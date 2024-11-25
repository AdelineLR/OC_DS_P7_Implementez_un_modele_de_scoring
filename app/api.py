# Projet P7 - Implémentez un modèle de scoring
# OPENCLASSROOMS - Parcours Data Scientist - Adeline Le Ray - 10/2024

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import os
import subprocess
import mlflow.sklearn
import time
import numpy as np
import pandas as pd
from joblib import load
import pickle

app = FastAPI()

# load model
pipeline_path = 'best_model_pipeline.joblib'
pipeline = load(pipeline_path)

# Extract the model and scaler from the pipeline
model = pipeline.named_steps['classification']
scaler = pipeline.named_steps['scaling']

# Load threshold value
with open('metric_dict.pkl', 'rb') as file:
    metric_dict = pickle.load(file)

threshold_value = metric_dict['threshold']

class PredictionInput(BaseModel):
    features: Dict[str, float]
        
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

        if predicted_class==0:
            status = 'Accepted'
        else:
            status = 'Rejected'
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "prediction": status,
        "predicted_class": predicted_class,
        "probabilities": {
            "class_0": probabilities[0][0],
            "class_1": probabilities[0][1]
        }
    }


if __name__ == '__main__':
    import uvicorn
    # Run the application
    uvicorn.run(app, host="0.0.0.0", port=8000)

