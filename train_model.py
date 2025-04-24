import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import mlflow
from pathlib import Path


MODEL_DIR = Path("model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_and_save_model():
    df = pd.read_csv("employee_data.csv")
    df['has_dependents'] = df['has_dependents'].map({'Yes': 1, 'No': 0})

    X = df.drop(columns=['employee_id', 'enrolled'])
    y = df['enrolled']

    label_encoders = {}
    categorical_cols = ['gender', 'marital_status', 'employment_type', 'region']

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        joblib.dump(le, MODEL_DIR / f"le_{col}.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
        # Define numeric columns
    numeric_cols = ['age', 'salary', 'tenure_years']

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")


    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)

    mlflow.set_experiment("Employee Enrollment Prediction")
    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print("Classification Report:\n", classification_report(y_test, y_pred))
        joblib.dump(model, MODEL_DIR / "enrollment_model.pkl")

        return {"accuracy": acc, "message": "Model and encoders saved to /model"}