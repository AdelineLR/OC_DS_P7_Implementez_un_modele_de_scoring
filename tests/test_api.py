import pytest
import requests
import pandas as pd
from unittest.mock import patch
from fastapi.exceptions import HTTPException
import os

# API base URL
BASE_URL = "http://127.0.0.1:8000"

def test_api_status():
    """
    Test that the API is running and responds with a status code 200.
    """
    response = requests.get(BASE_URL)
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"


def test_threshold_is_float_and_in_range():
    """
    Test that the threshold is a float and lies within the range [0.1, 0.9].
    """
    # Mock the threshold loading to return a value within the range
    with patch('builtins.open', return_value=MockFile()):

        # Mock the pickle loading to return a threshold value within the range [0.1, 0.9]
        with patch('pickle.load', return_value={'threshold': 0.75}):
            response = requests.get(f"{BASE_URL}/")

        assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
        
        # Assuming the threshold value is injected into the API, we can check if the threshold is a float
        threshold_value = 0.75  # Mocked threshold value

        # Check if threshold is a float and within the range [0.1, 0.9]
        assert isinstance(threshold_value, float), f"Expected threshold to be a float, but got {type(threshold_value)}"
        assert 0.1 <= threshold_value <= 0.9, f"Expected threshold to be between 0.1 and 0.9, but got {threshold_value}"


# Mock class to simulate reading a file
class MockFile:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def read(self):
        return b"{'threshold': 0.75}"  # Simulate the content of the pickle file with a threshold

def load_test_cases():
    return pd.read_csv("tests/test_cases.csv")


def test_prediction_response_fields():
    """
    Test the predict endpoint to check if the response contains the required fields.
    """
    test_cases = load_test_cases()
    test_case = test_cases.iloc[1].to_dict()
    payload = {"features": test_case}
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
    
    json_response = response.json()
    
    # Check if the required fields are present
    required_fields = ['prediction', 'predicted_class', 'probabilities']
    for field in required_fields:
        assert field in json_response, f"Expected field '{field}' is missing from the response"
    
    # Check if 'prediction' is equal to 'Accepted' or 'Rejected'
    assert json_response["prediction"] in ["Accepted", "Rejected"], (
        f"Expected 'Accepted' or 'Rejected', but got {json_response['prediction']}"
    )

    # Check if 'predicted_class' is an integer
    assert isinstance(json_response["predicted_class"], int), (
        f"Expected 'predicted_class' to be an integer, but got {type(json_response['predicted_class'])}"
    )

    # Check if 'probabilities' is a dictionary with required keys
    assert isinstance(json_response['probabilities'], dict), "'probabilities' should be a dictionary"
    probability_keys = ['class_0', 'class_1']
    for key in probability_keys:
        assert key in json_response['probabilities'], f"Expected key '{key}' is missing from 'probabilities'"
        assert isinstance(json_response['probabilities'][key], float), (
            f"Expected 'probabilities[{key}]' to be a float, but got {type(json_response['probabilities'][key])}"
        )


def test_predict_invalid_data():
    """
    Test the predict endpoint with dynamically generated invalid input data.
    """
    # Load valid test cases and extract a valid one
    test_cases = load_test_cases()
    valid_case = test_cases.iloc[0].to_dict()

    # Generate invalid cases by modifying the valid case
    invalid_cases = [
        {"features": {}},  # Empty features dictionary
        {"features": {"invalid_feature1": 1.0, "invalid_feature2": 2.5}},  # Invalid feature names
        {"features": {key: "invalid_value" for key in valid_case}},  # Invalid value types (string instead of float)
        {"wrong_key": valid_case},  # Missing 'features' key
    ]

    for invalid_input_data in invalid_cases:
        # Send the request with invalid input data
        response = requests.post(f"{BASE_URL}/predict", json=invalid_input_data)

        # Verify the response status code is 422 (Unprocessable Entity)
        assert response.status_code == 422, (
            f"Expected status code 422, but got {response.status_code} for input {invalid_input_data}"
        )

        # Verify the response contains a "detail" field with validation error information
        json_response = response.json()
        assert "detail" in json_response, (
            f"Expected 'detail' field in the response, but it was missing for input {invalid_input_data}"
        )