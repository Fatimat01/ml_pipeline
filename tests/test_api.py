import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_root():
    """
    Test GET method on root endpoint.
    Tests BOTH status code AND contents of response.
    """
    response = client.get("/")  
    # Test status code
    assert response.status_code == 200
    # Test contents of response object
    json_response = response.json()
    assert "message" in json_response
    assert json_response["message"] == "Census Income Classifier App"


def test_post_predict_income_below_50k():
    """
    Test POST for one possible inference: <=50K
    This tests the model predicting income <=50K.
    """
    data = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 226802,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States"
    }
    
    response = client.post("/predict", json=data)
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == "<=50K"


def test_post_predict_income_above_50k():
    """
    Test POST for the other possible inference: >50K
    This tests the model predicting income >50K.
    """
    data = {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }
    
    response = client.post("/predict", json=data)   
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == ">50K"