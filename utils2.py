import pandas as pd 
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x).rstrip('0').rstrip('.') if x != 0 else '0')
import numpy as np 
np.set_printoptions(suppress=True, precision=6)
import polars as pl 
pl.Config(set_fmt_float="full")
pl.Config(tbl_cols=1000)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os 
import seaborn as sns
np.random.seed(22)
import geopandas as gpd
import torch 
from torch.autograd import grad
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange, tqdm
from scipy.optimize import minimize 
import scipy.stats as stats
import json 
import itertools
import re
import ast
from PIL import Image
import os
import inspect
from scipy.optimize import minimize
from joblib import Parallel, delayed
import scipy
import numdifftools as nd

def is_missing(df):
    num=0
    for var in df.columns:
        c = df[var].isna().sum()
        if c>0:
            num+=1
    if num==0:
        print('No missing data in the database')
    else:
        print('Check your data, there is missing values...')

def get_param_names(func):
    import inspect
    import re

    # Get the source code of the function
    source = inspect.getsource(func)

    # Find the line where parameters are unpacked
    pattern = r'\(\s*([^\)]*?)\s*\)\s*=\s*params'
    match = re.search(pattern, source, re.DOTALL)

    if match:
        param_str = match.group(1)
        # Remove any line breaks and split the parameters
        param_names = [name.strip() for name in param_str.replace('\n', '').split(',')]
        # Remove any empty strings
        param_names = [name for name in param_names if name]
        return param_names
    else:
        raise ValueError("Could not find parameter unpacking in the function.")
    
def summary(init, hessian_inv, func, file_name='level0', save=False):
    param_names = get_param_names(func)
    coeff_values = init
    
    std_errors = np.sqrt(np.diag(hessian_inv))
    z_tests = coeff_values / std_errors
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_tests)))
    
    # Calcul des intervalles de confiance à 95%
    confidence_level = 0.95
    alpha = 1 - confidence_level
    z_critical = stats.norm.ppf(1 - alpha / 2)
    ci_lower = coeff_values - z_critical * std_errors
    ci_upper = coeff_values + z_critical * std_errors
    
    summary_df = pd.DataFrame({
        "Parameter": param_names,
        "Coeff. Value": coeff_values,
        "Std Error": std_errors,
        "Z-Test": z_tests,
        "P-Value": p_values,
        "95% CI Lower": ci_lower,
        "95% CI Upper": ci_upper
    })
    summary_df = summary_df.sort_values('Parameter', ascending=True)
    
    if save:
        summary_df
    
    return summary_df
    
def DisplayResults(func, params, data):    
    params_tensor = torch.tensor(params, dtype=torch.float32, requires_grad=True)
    loss = func(params_tensor, data, pytorch=True, grad=True)
    loss.backward()
    hessian = torch.autograd.functional.hessian(
        lambda p: func(p, data, pytorch=True, grad=True), 
        params_tensor
    )
    hessian_inv = torch.linalg.pinv(hessian)
    hessian_inv = hessian_inv.detach().numpy()

    BETAS = summary(params, hessian_inv, func)

    null_loglik = func(params, data, null_loglik=True)

    adj_mcfadden = 1 - (((-func(params, data)) - len(params)) / (-null_loglik))

    # summary(params, hessian_inv, func, file_name='level1', save=False)

    result = summary(params, hessian_inv, func, file_name='level1_MAN', save=False)
    metrics = pd.DataFrame({'Parameter':['Nb. of observation',
                                        'Log-Lik.',
                                        'Null Log-Lik.', 
                                        'Adj. McFadden',
                                        'AIC',
                                        'BIC'],
                'Coeff. Value':[func(params, data, df_length=True), 
                                np.round(func(params, data)), 
                                np.round(null_loglik.detach().numpy()), 
                                np.round(adj_mcfadden.detach().numpy(), 3),
                                np.round((-2*(-func(params, data)) + 2*len(params))),
                                np.round((-2*(-func(params, data)) + len(params)*np.log(func(params, data, df_length=True))))],
                'Std Error':['-', '-', '-', '-', '-', '-'],
                'Z-Test':['-', '-', '-', '-', '-', '-'],
                'P-Value':['-', '-', '-', '-', '-', '-']})

    table = pd.concat([result, metrics])

    pd.set_option('display.max_rows', 100)
    return table
    

def Optimize(func, 
             data,
             max_iter=5000,
             gtol=1e-3,
             initial_values=None, 
             save=False, 
             file_name=None,
             display_results=False):    

    def gradient(params_np, df):
        params_tensor = torch.tensor(params_np, dtype=torch.float32, requires_grad=True)
        loss = func(params_tensor, df, pytorch=True, grad=True)
        loss.backward()
        grad = params_tensor.grad.detach().numpy()
        return grad

    tqdm_bar = tqdm(total=max_iter, desc="Optimizing", position=0, leave=True)  
    def callbackF(xk, *args):
        obj_value = func(xk, data).item()  
        tqdm_bar.update(1)  
        tqdm_bar.set_postfix({'Objective Value': f"{obj_value:.5f}"})  
    callbackF.iter = 0 

    if initial_values is not None:
    ###### init values from another model:
        script_dir = os.getcwd()
        parameters_dir = os.path.join(script_dir, 'parameters')
        json_filename = os.path.join(parameters_dir, f'{initial_values}.json')
        with open(json_filename, "r") as json_file:
            json_loaded_parameters = json.load(json_file)
        func_params = get_param_names(func)
        filtered_dict = {param: json_loaded_parameters.get(param, 1) for param in func_params}
        init_values = torch.tensor(list(filtered_dict.values()), dtype=torch.float64)
    else:
    ###### init values at 0:
        init_values = torch.tensor([1]*2 + [0] * (len(get_param_names(func)) - 2), dtype=torch.float64)

    ###### optimization
    init = torch.nn.Parameter(init_values)
    params_estimated = minimize(func, init.detach().numpy(), 
                    args=(data, ), 
                    method='trust-constr', 
                    callback=callbackF,
                    jac=gradient,
                    options={'disp':False, 'gtol':gtol, 'maxiter':max_iter})
    if params_estimated.success:
        print('Convergence: True')
    else: 
        print('Convergence: False')

    if save:
        # Extraire les paramètres estimés
        estimated_parameters = torch.tensor(params_estimated.x)
        param_names = get_param_names(func)
        params_dict = {param: value for param, value in zip(param_names, estimated_parameters.tolist())}

        # Définir le chemin du dossier parameters
        parameters_dir = os.path.join(os.getcwd(), 'parameters')
        os.makedirs(parameters_dir, exist_ok=True)  # Crée le dossier s’il n’existe pas

        # Sauvegarde en JSON
        json_filename = os.path.join(parameters_dir, f'{file_name}.json')
        with open(json_filename, "w") as json_file:
            json.dump(params_dict, json_file, indent=4)

        # Sauvegarde en Excel
        excel_filename = os.path.join(parameters_dir, f'{file_name}.xlsx')
        excel_file = DisplayResults(func, estimated_parameters, data)
        excel_file.to_excel(excel_filename, index=False)
    if display_results:
        estimated_parameters = torch.tensor(params_estimated.x)
        return DisplayResults(func, estimated_parameters, data), estimated_parameters
    
    else: 
        estimated_parameters = torch.tensor(params_estimated.x)
        return estimated_parameters

def SaveParameters(func, params, excel=False, data=None, file_name=None):
    param_names = get_param_names(func)
    params_dict = {param: value for param, value in zip(param_names, params.tolist())}

    # Sauvegarde en JSON
    parameters_dir = os.path.join(os.getcwd(), 'parameters')
    json_filename = json_filename = os.path.join(parameters_dir, f'{file_name}.json')
    with open(json_filename, "w") as json_file:
        json.dump(params_dict, json_file, indent=4)

    if excel==True and data is not None:
        excel_filename = os.path.join(parameters_dir, f'{file_name}.xlsx')
        DisplayResults(func, params, data).to_excel(excel_filename, index=False) 

def LoadParameters(func, file_name):
        script_dir = os.getcwd()
        parameters_dir = os.path.join(script_dir, 'parameters')
        json_filename = os.path.join(parameters_dir, f'{file_name}.json')
        with open(json_filename, "r") as json_file:
            json_loaded_parameters = json.load(json_file)
        func_params = get_param_names(func)
        filtered_dict = {param: json_loaded_parameters.get(param, 1) for param in func_params}
        init_values = torch.tensor(list(filtered_dict.values()), dtype=torch.float64)

        return init_values

# -------- JOINT FUNCTIONS

def JointOptimize(func, 
             data,
             max_iter=5000,
             gtol=1e-3, 
             initial_values=None):    

    def gradient(params_np, df):
        params_tensor = torch.tensor(params_np, dtype=torch.float32, requires_grad=True)
        loss = func(params_tensor, df, pytorch=True, grad=True)
        loss.backward()
        grad = params_tensor.grad.detach().numpy()
        return grad

    tqdm_bar = tqdm(total=max_iter, desc="Optimizing", position=0, leave=True)  
    def callbackF(xk, *args):
        obj_value = func(xk, data).item()  
        tqdm_bar.update(1)  
        tqdm_bar.set_postfix({'Objective Value': f"{obj_value:.5f}"})  
    callbackF.iter = 0 

    ###### optimization
    init = torch.nn.Parameter(initial_values)
    params_estimated = minimize(func, init.detach().numpy(), 
                    args=(data, ), 
                    method='trust-constr', 
                    callback=callbackF,
                    jac=gradient,
                    options={'disp':False, 'gtol':gtol, 'maxiter':max_iter})
    if params_estimated.success:
        print('Convergence: True')
    else: 
        print('Convergence: False')

    return params_estimated.x

def JointSummary(init, 
                 hessian_inv,
                 level0,
                 level1,
                 level2,
                       ):
    param_names = (get_param_names(level0) 
                   + get_param_names(level1) 
                   + get_param_names(level2) 
                      )       
    coeff_values = init
    
    std_errors = np.sqrt(np.diag(hessian_inv))
    z_tests = coeff_values / std_errors
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_tests)))
    
    # Calcul des intervalles de confiance à 95%
    confidence_level = 0.95
    alpha = 1 - confidence_level
    z_critical = stats.norm.ppf(1 - alpha / 2)
    ci_lower = coeff_values - z_critical * std_errors
    ci_upper = coeff_values + z_critical * std_errors
    
    summary_df = pd.DataFrame({
        "Parameter": param_names,
        "Coeff. Value": coeff_values,
        "Std Error": std_errors,
        "Z-Test": z_tests,
        "P-Value": p_values,
        "95% CI Lower": ci_lower,
        "95% CI Upper": ci_upper
    })
    summary_df = summary_df.sort_values('Parameter', ascending=True)
    
    return summary_df

def JointDisplayResults(func, 
                        params, 
                        data,
                        level0,
                        level1,
                        level2,
                        ):    
    params_tensor = torch.tensor(params, dtype=torch.float32, requires_grad=True)
    loss = func(params_tensor, data, pytorch=True, grad=True)
    loss.backward()
    hessian = torch.autograd.functional.hessian(
        lambda p: func(p, data, pytorch=True, grad=True), 
        params_tensor
    )
    hessian_inv = torch.linalg.pinv(hessian)
    hessian_inv = hessian_inv.detach().numpy()

    BETAS = JointSummary(params, 
                 hessian_inv,
                 level0,
                 level1,
                 level2,
                       )

    null_loglik = func(params, data, null_loglik=True)

    adj_mcfadden = 1 - (((-func(params, data)) - len(params)) / (-null_loglik))

    result = JointSummary(params, 
                 hessian_inv,
                 level0,
                 level1,
                 level2,
                       )
    metrics = pd.DataFrame({'Parameter':['Nb. of observation',
                                        'Log-Lik.',
                                        'Null Log-Lik.', 
                                        'Adj. McFadden',
                                        'AIC',
                                        'BIC'],
                'Coeff. Value':[len(data), 
                                np.round(func(params, data)), 
                                np.round(null_loglik), 
                                np.round(adj_mcfadden, 3),
                                np.round((-2*(-func(params, data)) + 2*len(params))),
                                np.round((-2*(-func(params, data)) + len(params)*np.log(len(data))))],
                'Std Error':['-', '-', '-', '-', '-', '-'],
                'Z-Test':['-', '-', '-', '-', '-', '-'],
                'P-Value':['-', '-', '-', '-', '-', '-']})

    table = pd.concat([result, metrics])

    tables = {}

    for level in ['l0', 'l1', 'l2', 'l3']:

        if level=='l0':
            filtered_table_m = table[table['Parameter'].str.endswith(f'_m_{level}')]
            filtered_table_w = table[table['Parameter'].str.endswith(f'_w_{level}')]
            sigma = table[table['Parameter']=='sigma_l0']
            filtered_table = pd.concat([filtered_table_m, filtered_table_w], ignore_index=True)
            filtered_table = pd.concat([filtered_table, sigma], ignore_index=True)
            header_row = pd.DataFrame([{'Parameter': f'LEVEL_{level}'}])
            filtered_table = pd.concat([header_row, filtered_table], ignore_index=True)
            tables[f'{level}'] = filtered_table
        else:
            filtered_table = table[table['Parameter'].str.endswith(f'_{level}')]
            header_row = pd.DataFrame([{'Parameter': f'LEVEL_{level}'}])
            filtered_table = pd.concat([header_row, filtered_table], ignore_index=True)
            tables[f'{level}'] = filtered_table

    metrics = table[table['Parameter'].isin(['Log-Lik.', 'Null Log-Lik.', 'Adj. McFadden', 'AIC', 'BIC'])]

    final_table = pd.concat(tables, ignore_index=True)
    final_table = pd.concat([final_table, metrics], ignore_index=True)

    final_table

    return final_table

def JointSaveParameters(params,
                        level0,
                        level1,
                        level2,
                        excel=False, 
                        file_name=None,
                        table=None):
    
    param_names = (get_param_names(level0) 
                   + get_param_names(level1) 
                   + get_param_names(level2) 
                                           ) 
    
    params_dict = {param: value for param, value in zip(param_names, params.tolist())}

    # Sauvegarde en JSON
    parameters_dir = os.path.join(os.getcwd(), 'parameters')
    json_filename = json_filename = os.path.join(parameters_dir, f'{file_name}.json')
    with open(json_filename, "w") as json_file:
        json.dump(params_dict, json_file, indent=4)

    if excel==True:
        excel_filename = os.path.join(parameters_dir, f'{file_name}.xlsx')
        table.to_excel(excel_filename, index=False) 

def JointLoadParameters(file_name,
                        level0,
                        level1,
                        level2,
                              ):
    try:
        script_dir = os.getcwd()
        parameters_dir = os.path.join(script_dir, 'parameters')
        json_filename = os.path.join(parameters_dir, f'{file_name}.json')

        with open(json_filename, "r") as json_file:
            json_loaded_parameters = json.load(json_file)

        func_params = (get_param_names(level0) 
                    + get_param_names(level1) 
                    + get_param_names(level2) 
                                            ) 

        filtered_dict = {param: json_loaded_parameters.get(param, 1) for param in func_params}
        init_values = torch.tensor(list(filtered_dict.values()), dtype=torch.float64)

        return init_values

    except FileNotFoundError:
            print(f"Error:")
            print(f"The parameters that you try to load do not exist in the parameters folder.")
            print(f"You have to run the simultaneous model first with all_params and then save the results with JointSaveParameters.")
            print('Then you can load the saved parameters by using the filename that you put into JointSaveParameters.')

# ---------- PREPARE DATA FUNCTIONS 

def prepare_data_level0(df, year):
        
        tensor_vars = {
                col: torch.tensor(df[col].to_numpy(), dtype=torch.float64)
                for col in df.columns
                if not pd.api.types.is_object_dtype(df[col]) # filter to keep only integers and floats
        }

        if year>2016:
            men_idx = torch.where(torch.isin(tensor_vars['TRANS_m'], torch.tensor([2, 3, 4, 6])))[0]
            women_idx = torch.where(torch.isin(tensor_vars['TRANS_w'], torch.tensor([2, 3, 4, 6])))[0]
        else:
            men_idx = torch.where(torch.isin(tensor_vars['TRANS_m'], torch.tensor([2, 3, 5])))[0]
            women_idx = torch.where(torch.isin(tensor_vars['TRANS_w'], torch.tensor([2, 3, 5])))[0]


        all_idx = torch.where(~torch.isnan(tensor_vars['VOIT']))[0]

        return {
                "df": df,
                "vars": tensor_vars,
                "men_idx": men_idx,
                "women_idx": women_idx,
                "all_idx": all_idx
        }

def prepare_data_level1(df):

        tensor_vars = {
                col: torch.tensor(df[col].to_numpy(), dtype=torch.float64)
                for col in df.columns
                if not pd.api.types.is_object_dtype(df[col])
        }

        is_one_car_idx = torch.where(tensor_vars['VOIT'] == 1)[0]
        is_multi_car_idx = torch.where(tensor_vars['VOIT'] >= 2)[0]
        all_idx = torch.where(~torch.isnan(tensor_vars['VOIT']))[0]

        return {
                "df": df,
                "vars": tensor_vars,
                "is_one_car_idx": is_one_car_idx,
                "is_multi_car_idx": is_multi_car_idx,
                "all_idx": all_idx
        }

def prepare_data_level2(df):

        tensor_vars = {
                col: torch.tensor(df[col].to_numpy(), dtype=torch.float64)
                for col in df.columns
                if not pd.api.types.is_object_dtype(df[col])
        }

        at_least_one_car_idx = torch.where(tensor_vars['VOIT'] >= 1)[0]
        all_idx = torch.where(~torch.isnan(tensor_vars['VOIT']))[0]

        return {
                "df": df,
                "vars": tensor_vars,
                "at_least_one_car_idx": at_least_one_car_idx,
                "all_idx": all_idx
        }

def prepare_data_level3(df):

        tensor_vars = {
                col: torch.tensor(df[col].to_numpy(), dtype=torch.float64)
                for col in df.columns
                if not pd.api.types.is_object_dtype(df[col])
        }

        return {
                "df": df,
                "vars": tensor_vars
        }


# -- PREPARE DATA FOR SINGLE OR MONO ACTIVES 

def prepare_data_level0_mono(df, year):
        
        tensor_vars = {
                col: torch.tensor(df[col].to_numpy(), dtype=torch.float64)
                for col in df.columns
                if not pd.api.types.is_object_dtype(df[col]) # filter to keep only integers and floats
        }

        if year>2016:
            idx = torch.where(torch.isin(tensor_vars['TRANS'], torch.tensor([2, 3, 4, 6])))[0]
        else:
            idx = torch.where(torch.isin(tensor_vars['TRANS'], torch.tensor([2, 3, 5])))[0]

        all_idx = torch.where(~torch.isnan(tensor_vars['VOIT']))[0]

        return {
                "df": df,
                "vars": tensor_vars,
                "idx": idx,
                "all_idx": all_idx
        }

def prepare_data_level1_mono(df):

        tensor_vars = {
                col: torch.tensor(df[col].to_numpy(), dtype=torch.float64)
                for col in df.columns
                if not pd.api.types.is_object_dtype(df[col])
        }

        idx = torch.where(tensor_vars['VOIT'] >= 1)[0]
        all_idx = torch.where(~torch.isnan(tensor_vars['VOIT']))[0]

        return {
                "df": df,
                "vars": tensor_vars,
                "idx": idx,
                "all_idx": all_idx
        }

def prepare_data_level2_mono(df):

        tensor_vars = {
                col: torch.tensor(df[col].to_numpy(), dtype=torch.float64)
                for col in df.columns
                if not pd.api.types.is_object_dtype(df[col])
        }

        return {
                "df": df,
                "vars": tensor_vars
        }