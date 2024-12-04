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
