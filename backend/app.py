# CropGuard - FastAPI deployment
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from feature_engineering import add_features  # needed so joblib can load the model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_recommendation.pkl")

saved = joblib.load(MODEL_PATH)
model = saved["model"]
FEATURES = saved["features"]
RANGES = saved["input_ranges"]
THRESHOLD = saved["confidence_threshold"]

app = FastAPI(title="CropGuard - Crop Recommendation API")


class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


@app.get("/")
def home():
    return {"message": "CropGuard API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model": saved["model_name"]}


@app.post("/predict")
def predict(data: CropInput):
    values = data.model_dump() if hasattr(data, "model_dump") else data.dict()

    # 1. Out-of-range detection (compared with training data ranges)
    out_of_range = []
    for f in FEATURES:
        lo, hi = RANGES[f]["min"], RANGES[f]["max"]
        if values[f] < lo or values[f] > hi:
            out_of_range.append({
                "feature": f,
                "value": values[f],
                "allowed_min": lo,
                "allowed_max": hi
            })

    # 2. Model prediction + confidence
    X = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    probs = model.predict_proba(X)[0]
    classes = model.classes_
    order = probs.argsort()[::-1]
    top3 = [{"crop": str(classes[i]), "confidence": round(float(probs[i]), 4)}
            for i in order[:3]]
    best_crop = top3[0]["crop"]
    confidence = top3[0]["confidence"]

    # 3. Decision
    if out_of_range:
        status = "out_of_range"
        message = "One or more inputs are outside the range seen during training. Recommendation is not reliable."
        recommended = None
    elif confidence < THRESHOLD:
        status = "low_confidence"
        message = f"Model confidence is below {THRESHOLD}. Recommendation is uncertain."
        recommended = None
    else:
        status = "success"
        message = "Crop recommended with good confidence."
        recommended = best_crop

    return {
        "status": status,
        "message": message,
        "recommended_crop": recommended,
        "best_guess": best_crop,
        "confidence": confidence,
        "confidence_threshold": THRESHOLD,
        "top_3_predictions": top3,
        "out_of_range_features": out_of_range
    }
