# Put the code for your API here.

import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from ml.data import process_data
from ml.model import inference

# DVC setup for Heroku
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Initialize FastAPI app
app = FastAPI(
    title="Census Income Classifier API",
    description="API for predicting whether income exceeds $50K/yr based on census data",
    version="1.0.0"
)

# Load model and encoders
model = joblib.load("model/best_model.pkl")
encoder = joblib.load("model/encoder.pkl")
lb = joblib.load("model/label_binarizer.pkl")

# Define categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class CensusData(BaseModel):
    """
    Pydantic model for input data validation.
    
    Note: Using Field with alias to handle hyphens in column names.
    """
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")

    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionOutput(BaseModel):
    """Output model for predictions."""
    prediction: str


@app.get("/")
async def root():
    """
    Welcome endpoint.
    
    Returns a greeting message.
    """
    return {"message": "Welcome to the Census Income Classifier API! Use POST /predict to make predictions."}


@app.post("/predict", response_model=PredictionOutput)
async def predict(data: CensusData):
    """
    Predict income class based on census data.
    
    Parameters:
    - **data**: Census data following the CensusData schema
    
    Returns:
    - **prediction**: Either "<=50K" or ">50K"
    """
    # Convert input to DataFrame with correct column names
    input_dict = {
        "age": [data.age],
        "workclass": [data.workclass],
        "fnlgt": [data.fnlgt],
        "education": [data.education],
        "education-num": [data.education_num],
        "marital-status": [data.marital_status],
        "occupation": [data.occupation],
        "relationship": [data.relationship],
        "race": [data.race],
        "sex": [data.sex],
        "capital-gain": [data.capital_gain],
        "capital-loss": [data.capital_loss],
        "hours-per-week": [data.hours_per_week],
        "native-country": [data.native_country]
    }
    
    input_df = pd.DataFrame(input_dict)
    
    # Process the data
    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    # Make prediction
    pred = inference(model, X)
    
    # Convert prediction to label
    prediction_label = lb.inverse_transform(pred)[0]
    
    return {"prediction": prediction_label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)