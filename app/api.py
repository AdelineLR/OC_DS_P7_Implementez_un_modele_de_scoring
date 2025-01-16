# Projet P7 - Implémentez un modèle de scoring
# OPENCLASSROOMS - Parcours Data Scientist - Adeline Le Ray - 10/2024

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import pandas as pd
from joblib import load
import pickle
import shap

app = FastAPI()

# Model loading
pipeline_path = 'app/best_model_pipeline.joblib'
pipeline = load(pipeline_path)

# Extract model and scaler
model = pipeline.named_steps['classification']
scaler = pipeline.named_steps['scaling']

# Threshold loading
with open('app/metric_dict.pkl', 'rb') as file:
    metric_dict = pickle.load(file)
    threshold_value = metric_dict['threshold']

# Shap Explainer loading
explainer_path = 'app/explainer.joblib'
explainer = load(explainer_path)

class PredictionInput(BaseModel):
    features: Dict[str, float]
        
@app.get("/")
async def root():
    return {"message": "Welcome to Prêt à Dépenser - Scoring API"}

@app.post('/predict')
def predict(input_data: PredictionInput):
    """Predict using the loaded model."""
    try:     
        # Ensure the 'features' dictionary is not empty
        if not input_data.features:
            raise HTTPException(status_code=422, detail="The 'features' dictionary cannot be empty.")
        
        # Ensure the features names are valid
        valid_feature_names = ['EXT_SOURCE_2', 'PAYMENT_RATE', 'EXT_SOURCE_3', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
       'ANNUITY_INCOME_PERCENT', 'INSTAL_DBD_MEAN', 'DAYS_LAST_PHONE_CHANGE',
       'AMT_ANNUITY', 'ACTIVE_DAYS_CREDIT_UPDATE_MEAN',
       'REGION_POPULATION_RELATIVE', 'INSTAL_DAYS_ENTRY_PAYMENT_MAX',
       'CLOSED_DAYS_CREDIT_MAX', 'ACTIVE_DAYS_CREDIT_ENDDATE_MIN',
       'INSTAL_AMT_PAYMENT_MIN', 'PREV_APP_CREDIT_PERC_VAR',
       'BURO_DAYS_CREDIT_VAR', 'INSTAL_DBD_SUM', 'INSTAL_DBD_MAX',
       'CLOSED_DAYS_CREDIT_ENDDATE_MAX', 'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN',
       'INCOME_PER_PERSON', 'ACTIVE_DAYS_CREDIT_MAX',
       'CLOSED_AMT_CREDIT_SUM_MEAN', 'PREV_HOUR_APPR_PROCESS_START_MEAN',
       'INSTAL_DAYS_ENTRY_PAYMENT_MEAN',
       'POS_NAME_CONTRACT_STATUS_Active_MEAN', 'TOTALAREA_MODE',
       'CLOSED_DAYS_CREDIT_UPDATE_MEAN', 'ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN',
       'BURO_DAYS_CREDIT_MEAN', 'PREV_CNT_PAYMENT_MEAN',
       'INSTAL_DAYS_ENTRY_PAYMENT_SUM', 'CLOSED_AMT_CREDIT_SUM_SUM',
       'INSTAL_AMT_PAYMENT_MEAN', 'PREV_APP_CREDIT_PERC_MEAN',
       'POS_MONTHS_BALANCE_SIZE', 'INSTAL_DPD_MEAN', 'PREV_AMT_ANNUITY_MIN',
       'PREV_AMT_ANNUITY_MEAN', 'PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN',
       'BURO_DAYS_CREDIT_ENDDATE_MIN', 'HOUR_APPR_PROCESS_START',
       'INSTAL_AMT_INSTALMENT_MAX', 'INSTAL_PAYMENT_PERC_VAR',
       'PREV_NAME_YIELD_GROUP_middle_MEAN', 'PREV_RATE_DOWN_PAYMENT_MEAN',
       'APPROVED_AMT_DOWN_PAYMENT_MAX'
       ]

        if not all(key in valid_feature_names for key in input_data.features.keys()):
            raise HTTPException(status_code=422, detail="Invalid feature names.")
                
        # Process input data
        features = input_data.features
        data = pd.DataFrame([features])

        # Scale and predict
        data_scaled = scaler.transform(data)
        probabilities = model.predict_proba(data_scaled)
        predicted_class = int(probabilities[0][1] >= threshold_value)

        if predicted_class == 0:
            status = 'Accepted'
        else:
            status = 'Rejected'

        # Local feature importance - shap values
        shap_value = explainer(data_scaled.reshape(1, -1))
        shap_value.feature_name = features

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        # Catch unexpected errors and return a meaningful error message
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    return {
        "prediction": status,
        "predicted_class": predicted_class,
        "probabilities": {
            "class_0": probabilities[0][0],
            "class_1": probabilities[0][1]
        },
        "shap_values" : shap_value
    }



if __name__ == '__main__':
    import uvicorn
    # Run the application
    uvicorn.run(app, host="0.0.0.0", port=8000)

