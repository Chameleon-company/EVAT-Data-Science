from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load("ev_model.pkl")

FEATURES = [
    "Year",
    "SHAPE_Length",
    "dist_to_nearest_ev_m",
    "ev_within_500m",
    "avg_temp",
    "total_prcp",
]

class StationFeatures(BaseModel):
    Year: int
    SHAPE_Length: float
    dist_to_nearest_ev_m: float
    ev_within_500m: int
    avg_temp: float
    total_prcp: float
    
app = FastAPI(title="EV Traffic & Weather Model API")

@app.get("/")
def root():
    return {"message": "EV model API is running"}

@app.post("/predict")
def predict(features: StationFeatures):
    data = pd.DataFrame([[getattr(features, f) for f in FEATURES]],
                        columns=FEATURES)

    pred = model.predict(data)[0]

    return {"prediction": float(pred)}