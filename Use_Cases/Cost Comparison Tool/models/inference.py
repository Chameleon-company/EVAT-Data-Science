from pathlib import Path
import joblib
import pandas as pd

EV_KWH_PER_KM = 0.15
# model file sits alongside this module
MODEL_PATH = Path(__file__).resolve().parent / "best_model.joblib"
_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def build_features(payload: dict) -> pd.DataFrame:
    # payload keys: distance_km, electricity_price_per_kwh, ice_eff_l_per_100km, petrol_price_per_l
    return pd.DataFrame([{
        "distance_km": payload["distance_km"],
        "electricity_price_per_kwh": payload["electricity_price_per_kwh"],
        "ice_eff_l_per_100km": payload["ice_eff_l_per_100km"],
        "petrol_price_per_l": payload["petrol_price_per_l"],
        "fuel_cost_per_km": (payload["ice_eff_l_per_100km"] / 100.0) * payload["petrol_price_per_l"],
        "ev_cost_per_km": payload["electricity_price_per_kwh"] * EV_KWH_PER_KM,
        "distance_x_petrol": payload["distance_km"] * payload["petrol_price_per_l"],
        "distance_x_elec": payload["distance_km"] * payload["electricity_price_per_kwh"],
        "eff_ratio": payload["ice_eff_l_per_100km"] / (EV_KWH_PER_KM * 100),
    }])

def predict_cost(payload: dict) -> float:
    model = load_model()
    feats = build_features(payload)
    return float(model.predict(feats)[0])