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
![image](https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png)

## Structure du repository
![structure_folder](https://github.com/user-attachments/assets/aa9b52e6-c6f3-40af-bc16-f6165f2b4887)

## Workflow modélisation
![modelisation_workflow](https://github.com/user-attachments/assets/27d858e8-1c0c-4486-bfe5-e7ed7773fab2)

## Pipeline de déploiement
![pipeline](https://github.com/user-attachments/assets/33e80b0d-1017-430b-b323-60210dd51e95)
