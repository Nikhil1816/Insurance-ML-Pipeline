from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
import os
import uvicorn
from train_model import train_and_save_model

app = FastAPI()
MODEL_DIR = Path("model")

class EmployeeInput(BaseModel):
    age: int
    gender: str
    marital_status: str
    salary: float
    employment_type: str
    region: str
    has_dependents: str
    tenure_years: float
from pydantic import BaseModel
from typing import List

class EmployeeBatchRequest(BaseModel):
    data: List[EmployeeInput]

model = None
scaler = None
label_encoders = {}

if (MODEL_DIR / "enrollment_model.pkl").exists():
    model = joblib.load(MODEL_DIR / "enrollment_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    label_encoders = {
        col: joblib.load(MODEL_DIR / f"le_{col}.pkl")
        for col in ["gender", "marital_status", "employment_type", "region"]
    }

@app.post("/predict_batch")
def predict_batch(batch: EmployeeBatchRequest):
    global model, label_encoders, scaler

    if model is None or scaler is None or len(label_encoders) == 0:
        return {"error": "Model not trained. Please POST to /train first."}

    df = pd.DataFrame([record.dict() for record in batch.data])
    df['has_dependents'] = df['has_dependents'].map({'Yes': 1, 'No': 0})

    for col, encoder in label_encoders.items():
        df[col] = encoder.transform(df[col])

    df[['age', 'salary', 'tenure_years']] = scaler.transform(df[['age', 'salary', 'tenure_years']])

    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    return [
        {
            "enrolled": int(pred),
            "probability": round(prob, 3)
        }
        for pred, prob in zip(predictions, probabilities)
    ]


@app.post("/train")
def train_model():
    global model, label_encoders, scaler
    results = train_and_save_model()
    model = joblib.load(MODEL_DIR / "enrollment_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    label_encoders = {
        col: joblib.load(MODEL_DIR / f"le_{col}.pkl")
        for col in ["gender", "marital_status", "employment_type", "region"]
    }
    return results

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
