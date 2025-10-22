"""
Script to POST to live Heroku API.
"""

import requests

# heroku app URL
API_URL = "https://census-income-classifier-688c084c733a.herokuapp.com/"  

# Test data
data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

# POST request
response = requests.post(f"{API_URL}/predict", json=data)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
