"""
Script to POST to live Heroku API.
Replace YOUR_HEROKU_APP_URL with your actual app URL.
"""
import requests

# Replace with your Heroku app URL
API_URL = "YOUR_HEROKU_APP_URL"  # e.g., "https://your-app.herokuapp.com"

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