import os
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from ml.data import process_data
from ml.model import inference

# DVC setup for Heroku
# if "DYNO" in os.environ and os.path.isdir(".dvc"):
#     os.system("dvc config core.no_scm true")
#     if os.system("dvc pull") != 0:
#         exit("dvc pull failed")
#     os.system("rm -r .dvc .apt/usr/lib/dvc")

# Initialize FastAPI app
app = FastAPI()

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
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        json_schema_extra = {
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


@app.get("/")
def get_root():
    """Welcome endpoint"""
    return {"message": "Census Income Classifier App"}


@app.post("/predict")
def predict(data: CensusData):
    """Predict income class"""
    try:
        # Convert to DataFrame
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
        prediction_label = lb.inverse_transform(pred)[0]
        
        return {"prediction": prediction_label}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )