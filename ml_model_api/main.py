import joblib
import pandas as pd
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Any

# Setup
app = FastAPI(title="Churn Model API", version="1.0")

# Load Model
api_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(api_dir, 'churn_model_v1.pkl')
model = None


@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(model_path)
        print("--- Model loaded successfully ---")
    except FileNotFoundError:
        print(f"---!! ERROR: Model file not found at {model_path} !! ---")
    except Exception as e:
        print(f"---!! ERROR: Could not load model: {e} !! ---")


# Data Validation
# Use Field(alias=...) for any names that the generator gets wrong.

class CustomerData(BaseModel):
    Tenure_Months: int = Field(alias="Tenure Months")
    Monthly_Charges: float = Field(alias="Monthly Charges")
    Total_Charges: float = Field(alias="Total Charges")
    Gender: str
    Senior_Citizen: str = Field(alias="Senior Citizen")
    Partner: str
    Dependents: str
    Phone_Service: str = Field(alias="Phone Service")
    Multiple_Lines: str = Field(alias="Multiple Lines")
    Internet_Service: str = Field(alias="Internet Service")
    Online_Security: str = Field(alias="Online Security")
    Online_Backup: str = Field(alias="Online Backup")
    Device_Protection: str = Field(alias="Device Protection")
    Tech_Support: str = Field(alias="Tech Support")
    Streaming_TV: str = Field(alias="Streaming TV")
    Streaming_Movies: str = Field(alias="Streaming Movies")
    Contract: str
    Paperless_Billing: str = Field(alias="Paperless Billing")
    Payment_Method: str = Field(alias="Payment Method")

    class Config:
        # Remove the alias_generator, but keep populate_by_name = True
        populate_by_name = True


# Define the output data structure
class PredictionOut(BaseModel):
    customer_id: Any
    churn_probability: float
    risk_level: str


# Create API endpoints

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Churn Model API is running."}


@app.post("/predict", response_model=PredictionOut)
def predict_churn(customer_id: Any, features: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        # Convert the Pydantic model to a dictionary
        data = features.model_dump(by_alias=True)

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame([data])

        # Get probability scores
        proba = model.predict_proba(df)
        churn_score = float(proba[0, 1])

        # Define risk levels
        risk_level = "Low"
        if churn_score >= 0.75:
            risk_level = "High"
        elif churn_score >= 0.40:
            risk_level = "Medium"

        # Return the response
        return {
            "customer_id": customer_id,
            "churn_probability": churn_score,
            "risk_level": risk_level
        }

    except Exception as e:
        print(f"--- Prediction Error ---")
        print(f"Data received: {data}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")