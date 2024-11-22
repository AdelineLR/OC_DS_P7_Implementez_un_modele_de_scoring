# Projet P7 - Implémentez un modèle de scoring
# OPENCLASSROOMS - Parcours Data Scientist - Adeline Le Ray - 10/2024

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import mlflow
from mlflow.tracking import MlflowClient
import os
import subprocess
import mlflow.sklearn
import time
import numpy as np
import pandas as pd

app = FastAPI()

# parameters
EXPERIMENT_NAME = 'OC_DS_P7_Scoring'
MODEL_NAME = "model_optimisation_custom_cost_2_grid_class_weight_None_lightgbm_classifier"


class PredictionInput(BaseModel):
    features: Dict[str, float]
        
def setup_mlflow(experiment_name):
    # Get the MLflow tracking URI from the .env file, default to SQLite db
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    # Launch MLflow UI if not already running
    print(f"Launching MLflow UI at {mlflow_uri}...")
   
    # Set the MLflow tracking URI
    mlflow.set_tracking_uri('http://localhost:5000')
    
    # Set up experiment
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI set to {mlflow_uri} and experiment set to {experiment_name}")

def load_model_scaler(model_name):
    """!
    @brief: Load the latest version of a model from the MLflow model registry.
    @param model_name: str, the name of the model registered in MLflow.
    @return: Loaded model instance and scaler.
    """
    client = MlflowClient()
    
    # Load the model pipeline using the alias directly
    model_uri = f"models:/{model_name}@champion"
    pipeline = mlflow.sklearn.load_model(model_uri)
    
    # Get run_id 
    model_version = client.get_model_version_by_alias(model_name, "champion")
    run_id = model_version.run_id

    # Access the metrics from the run
    run = client.get_run(model_version.run_id)
    threshold_value = run.data.metrics.get("threshold_optimal")

    # Extract the model and scaler from the pipeline
    model = pipeline.named_steps['classification']
    scaler = pipeline.named_steps['scaling']
    
    return model, scaler, threshold_value

# Load the model and scaler when the app starts
setup_mlflow(EXPERIMENT_NAME)
time.sleep(5)
model, scaler, threshold_value = load_model_scaler(MODEL_NAME)

@app.get("/")
async def root():
    return {"message": "Welcome to Prêt à Dépenser - Scoring API"}

@app.post('/predict')
def predict(input_data: PredictionInput):
    """Predict using the loaded model."""
    
    try:
        features = input_data.features
        data = pd.DataFrame([features])

        # Scale and predict
        data_scaled = scaler.transform(data)
        probabilities = model.predict_proba(data_scaled)
        predicted_class = int(probabilities[0][1] >= threshold_value)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"prediction": predicted_class, "probabilities": probabilities[0][1]}


if __name__ == '__main__':
    import uvicorn
    # Run the application
    uvicorn.run(app, host="0.0.0.0", port=8000)

