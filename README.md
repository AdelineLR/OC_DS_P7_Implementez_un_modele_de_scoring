# Implémentez un modele de scoring

_Projet réalisé dans le cadre de la formation Data Scientist d'OpenClassrooms (Projet n°7 - Oct. 2024)_

## Mission
"Prêt à dépenser" est une société financière qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.
L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.)

La mission consiste à :
1. Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
2. Analyser les features qui contribuent le plus au modèle, d’une manière générale (feature importance globale) et au niveau d’un client (feature importance locale), afin, dans un soucis de transparence, de permettre à un chargé d’études de mieux comprendre le score attribué par le modèle.
3. Mettre en production le modèle de scoring de prédiction à l’aide d’une API et réaliser une interface de test de cette API.
Mettre en œuvre une approche globale MLOps de bout en bout, du tracking des expérimentations à l’analyse en production du data drift.

* [Données](https://www.kaggle.com/c/home-credit-default-risk/data)

## Structure du repository
OC_DS_P7_Implementez_un_modele_de_scoring/
├── .github/worflows/
│   └── deploy.yml                                              <- GitHub actions
├── app/                                                        <- API code
│   ├── api.py
│   ├── best_model_pipeline.joblib
│   ├── metric_dict.pkl
│   └── requirements.txt
├── modelisation/                                               <- Exploratory Data Analysis, Modelisation, Data Drit
│   ├── LeRay_Adeline_2.1_Notebook_EDA_102024.ipynb
│   ├── LeRay_Adeline_2.2_Notebook_modelisation_102024.ipynb
│   ├── LeRay_Adeline_2_3_preprocess_functions_102024.py
│   ├── LeRay_Adeline_2_4_modelisation_functions_102024.py
│   └── data_drift_report.html
├── tests/                                                      <- API unit tests   
│   ├── test_api.py
│   └── test_cases.csv 
├── .dockerignore                                               
├── .gitignore
├── Dockerfile                                                  <- Code to build API docker image
├── README.md
└── test_api.ipynb                                              <- Notebook for API testing


## Workflow modélisation

 ![alt text](modelisation_workflow.png)

## Pipeline de déploiement

![alt text](pipeline.png) 