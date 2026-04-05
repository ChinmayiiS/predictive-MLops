from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model/model.pkl")

@app.get("/")
def home():
    return {"message": "Printer Failure Prediction API"}

@app.post("/predict")
def predict(data: dict):
    features = np.array(list(data.values())).reshape(1, -1)

    prob = model.predict_proba(features)[0][1]
    prediction = int(prob > 0.5)

    return {
        "failure_prediction": prediction,
        "failure_probability": float(prob)
    }
