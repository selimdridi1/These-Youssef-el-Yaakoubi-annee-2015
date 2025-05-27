# 🚦 Modélisation des Comportements de Mobilité (Nested Logit)  
*Projet de Conduite de Projet - Master 1 Université de Strasbourg*


 Modélisation des célibataires mono-actifs  des couples bi-actifs

---

## 📌 Objectif du Projet

Ce projet a pour but d'analyser et de modéliser les **décisions de mobilité des individus en milieu urbain**, en utilisant des données issues du recensement INSEE (2015).  
Il s'appuie sur des modèles économétriques avancés de type **Nested Logit** pour capter la hiérarchie des décisions : du choix modal à la possession d’un véhicule.

Nous étudions trois populations distinctes :

1. **Célibataires **  
2. **Mono-actifs** 
3. **Couples bi-actifs**  

---

## 🧠 Méthodologie : Nested Logit Hiérarchique

Chaque groupe est modélisé à travers des niveaux imbriqués (Nested Logit), reflétant la structure des choix :

### Célibataires et Mono-Actifs
- **Niveau 0** : Choix modal (TC / 2 roues / Marche)
- **Niveau 1** : Arbitrage entre voiture et alternatives
- **Niveau 2** : Décision d'achat d’un véhicule

### Couples Bi-Actifs
- **Niveau 0** : Choix modal individuel (par conjoint)
- **Niveau 1** : Configuration modale conjointe (voiture, TC, marche...)
- **Niveau 2** : Achat du second véhicule
- **Niveau 3** : Achat du premier véhicule

---

## 🧾 Données

Les données proviennent du **recensement de la population 2015** en Île-de-France.  
Elles sont stockées au format `.parquet` et traitées à l’aide de `pandas`, `polars` et `pyarrow`.






