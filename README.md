# Implémentez un modele de scoring

_Projet réalisé dans le cadre de la formation Data Scientist d'OpenClassrooms (Projet n°7 - Oct. 2024)_

## Contexte
Une fintech souhaite automatiser l’évaluation du risque de défaut de crédit tout en garantissant transparence et accessibilité pour ses clients et collaborateurs.

## Mission
Développer un modèle de scoring crédit fiable, interprétable et accessible via un dashboard cloud, en s’appuyant sur des données variées (comportementales, externes…).

## Actions
1. Conception d’un modèle de classification prédictif du risque de défaut (ML supervisé, Cross-Validation et optimisation des hyperparamètres).
2. Analyse des variables contributives globales et locales (feature importance, SHAP).
3. Déploiement continu du modèle dans le cloud via une API REST (FastAPI, Docker, AWS) en utilisant des github actions
4. Conception d’un dashboard interactif (Streamlit) pour visualiser le score, son interprétation, et comparer un client à des groupes similaires (voir repository [OC_DS_Dashboard_scoring_app](https://github.com/AdelineLR/OC_DS_Dashboard_scoring_app)).
5. Intégration de bonnes pratiques MLOps (experiment tracking – Mlflow, versionning-Git, test - pytest, data drift - evidently, accessibilité WCAG).

## Résultat 
Un système complet de scoring crédit, transparent et accessible, déployé sur le cloud, utilisé par les chargés de clientèle pour justifier les décisions de crédit.

## [Données](https://www.kaggle.com/c/home-credit-default-risk/data)
![image](https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png)

## Structure du repository
![structure_folder](https://github.com/user-attachments/assets/aa9b52e6-c6f3-40af-bc16-f6165f2b4887)

## Workflow modélisation
![modelisation_workflow](https://github.com/user-attachments/assets/27d858e8-1c0c-4486-bfe5-e7ed7773fab2)

## Pipeline de déploiement
![pipeline](https://github.com/user-attachments/assets/33e80b0d-1017-430b-b323-60210dd51e95)
