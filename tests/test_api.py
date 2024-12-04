import pytest
import requests

# API base URL
BASE_URL = "http://100.26.100.225"

def test_api_status():
    """
    Test that the API is running and responds with a status code 200.
    """
    response = requests.get(BASE_URL)
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"

def load_test_cases():
    return pd.read_csv("test_cases.csv")

def test_prediction_positive_case():
    """
    Test the predict endpoint with a positive case (class 0).
    """
    test_cases = load_test_cases()
    positive_case = test_cases.iloc[0].to_dict()  
    payload = {"features": positive_case}
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
    
    json_response = response.json()
    assert json_response["predicted_class"] == 0, f"Expected class 0, but got {json_response['predicted_class']}"

def test_prediction_negative_case():
    """
    Test the predict endpoint with a negative case (class 1).
    """
    test_cases = load_test_cases()
    negative_case = test_cases.iloc[1].to_dict()
    payload = {"features": negative_case}
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
    
    json_response = response.json()
    assert json_response["predicted_class"] == 1, f"Expected class 1, but got {json_response['predicted_class']}"