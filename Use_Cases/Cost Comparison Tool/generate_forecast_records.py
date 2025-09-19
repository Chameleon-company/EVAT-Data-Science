#!/usr/bin/env python3
"""
Train multiple models (as in model.py), select the best by R², save it,
then generate large-scale forecast rows (20k–60k+) to CSV in batches.

Inputs:
  - dummy_data.csv  (trip-level dummy dataset)

Outputs:
  - models/best_model.joblib
  - forecast_output.csv

Run:
  python generate_forecasts_with_best_model.py
"""

import os
import math
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from joblib import dump

# ----------------------------
# Configuration (tune here)
# ----------------------------
DATA_PATH = "dummy_data.csv"
OUTPUT_CSV = "forecast_output.csv"
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "best_model.joblib"

START_YEAR = 2025
END_YEAR   = 2035            # inclusive
MONTHS     = list(range(1, 13))

# Base prices (AUD)
BASE_PETROL = 2.00           # $/L
BASE_ELEC   = 0.30           # $/kWh

# Annual growth scenarios
PETROL_GROWTHS = [0.02, 0.05, 0.08]       # low/med/high
ELEC_GROWTHS   = [0.01, 0.02, 0.035]      # low/med/high

# Trip profiles
DISTANCES = [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000]  # km
ICE_EFFS  = [5.5, 6.5, 7.5, 8.5, 10.0]                          # L/100km

# Batch write size
BATCH_ROWS = 50_000

# Assumed EV consumption (kWh/km) for engineered features
EV_KWH_PER_KM = 0.15

# ----------------------------
# 1) Load & clean data
# ----------------------------
df = pd.read_csv(DATA_PATH)

# Conservative clipping to remove pathologies
df["petrol_price_per_l"] = df["petrol_price_per_l"].clip(lower=0, upper=5.0)
df["electricity_price_per_kwh"] = df["electricity_price_per_kwh"].clip(lower=0, upper=2.0)
df["distance_km"] = df["distance_km"].clip(lower=0, upper=df["distance_km"].quantile(0.99))
df["ice_eff_l_per_100km"] = df["ice_eff_l_per_100km"].clip(lower=0)

# Target (winsorize tails for stability)
y_raw = df["savings_ice_minus_ev"].copy()
q1, q99 = y_raw.quantile([0.01, 0.99])
y = y_raw.clip(q1, q99)

# ----------------------------
# 2) Feature engineering (same as model.py)
# ----------------------------
df["fuel_cost_per_km"] = (df["ice_eff_l_per_100km"] / 100.0) * df["petrol_price_per_l"]
df["ev_cost_per_km"] = df["electricity_price_per_kwh"] * EV_KWH_PER_KM
df["distance_x_petrol"] = df["distance_km"] * df["petrol_price_per_l"]
df["distance_x_elec"] = df["distance_km"] * df["electricity_price_per_kwh"]
df["eff_ratio"] = df["ice_eff_l_per_100km"] / (EV_KWH_PER_KM * 100)

FEATURES = [
    "distance_km",
    "electricity_price_per_kwh",
    "ice_eff_l_per_100km",
    "petrol_price_per_l",
    "fuel_cost_per_km",
    "ev_cost_per_km",
    "distance_x_petrol",
    "distance_x_elec",
    "eff_ratio",
]
X = df[FEATURES].copy()

# ----------------------------
# 3) Split & define model grid
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
preprocess = ColumnTransformer(
    transformers=[("num", numeric_transformer, FEATURES)],
    remainder="drop",
)

models = {
    "LinearRegression": Pipeline([
        ("preprocess", preprocess),
        ("reg", LinearRegression())
    ]),
    "Ridge(alpha=1.0)": Pipeline([
        ("preprocess", preprocess),
        ("reg", Ridge(alpha=1.0, random_state=42))
    ]),
    "Poly2 + Ridge": Pipeline([
        ("poly_prep",
         ColumnTransformer(transformers=[
             ("poly", Pipeline([
                 ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                 ("scaler", StandardScaler())
             ]), FEATURES),
         ], remainder="drop")),
        ("reg", Ridge(alpha=1.0, random_state=42)),
    ]),
    "RandomForest": Pipeline([
        ("preprocess", "passthrough"),
        ("reg", RandomForestRegressor(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ))
    ]),
    "GradientBoosting": Pipeline([
        ("preprocess", "passthrough"),
        ("reg", GradientBoostingRegressor(
            n_estimators=350, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=42
        ))
    ]),
}

# ----------------------------
# 4) Train, score, pick best
# ----------------------------
def evaluate(y_true, y_pred, label):
    return {
        "Model": label,
        "R2": r2_score(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAE": mean_absolute_error(y_true, y_pred),
    }

results = []
trained = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    results.append(evaluate(y_test, y_pred, name))
    trained[name] = pipe

score_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\n=== Model Scorecard ===")
print(score_df.to_string(index=False))

best_name = score_df.iloc[0]["Model"]
best_model = trained[best_name]
print(f"\nSelected best model: {best_name}")

# Save the best model
dump(best_model, MODEL_PATH)
print(f"Saved best model to: {MODEL_PATH.resolve()}")

# ----------------------------
# 5) Forecast generation
# ----------------------------
def yearly_price(base, growth, years_since_start):
    return base * ((1.0 + growth) ** years_since_start)

def make_features_frame(distance_km, elec_price, ice_eff, petrol_price):
    """1-row DataFrame with engineered features, in the same order as training."""
    return pd.DataFrame([{
        "distance_km": distance_km,
        "electricity_price_per_kwh": elec_price,
        "ice_eff_l_per_100km": ice_eff,
        "petrol_price_per_l": petrol_price,
        "fuel_cost_per_km": (ice_eff / 100.0) * petrol_price,
        "ev_cost_per_km": elec_price * EV_KWH_PER_KM,
        "distance_x_petrol": distance_km * petrol_price,
        "distance_x_elec": distance_km * elec_price,
        "eff_ratio": ice_eff / (EV_KWH_PER_KM * 100),
    }], columns=FEATURES)

def scenario_rows():
    """Yield forecast rows across years × months × growths × distance × ICE eff."""
    years = list(range(START_YEAR, END_YEAR + 1))
    for year, month, pg, eg, dist, ice_eff in product(
        years, MONTHS, PETROL_GROWTHS, ELEC_GROWTHS, DISTANCES, ICE_EFFS
    ):
        years_since_start = year - START_YEAR
        petrol_price = yearly_price(BASE_PETROL, pg, years_since_start)
        elec_price   = yearly_price(BASE_ELEC,   eg, years_since_start)

        feats = make_features_frame(dist, elec_price, ice_eff, petrol_price)
        pred_savings = float(best_model.predict(feats)[0])

        # also provide simple rule-based EV/ICE costs for context
        rule_ev_cost  = dist * elec_price * EV_KWH_PER_KM
        rule_ice_cost = (ice_eff / 100.0) * dist * petrol_price

        yield {
            "year": year,
            "month": month,
            "petrol_growth": pg,
            "electricity_growth": eg,
            "distance_km": dist,
            "ice_eff_l_per_100km": ice_eff,
            "petrol_price_per_l": round(petrol_price, 4),
            "electricity_price_per_kwh": round(elec_price, 4),
            "predicted_savings": round(pred_savings, 2),
            "rule_ev_cost": round(rule_ev_cost, 2),
            "rule_ice_cost": round(rule_ice_cost, 2),
        }

# Estimate rows
total_rows = (
    (END_YEAR - START_YEAR + 1)
    * len(MONTHS) * len(PETROL_GROWTHS) * len(ELEC_GROWTHS)
    * len(DISTANCES) * len(ICE_EFFS)
)
print(f"\nPlanned forecast rows: ~{total_rows:,}")

# Stream to CSV in batches
if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)

cols = [
    "year","month","petrol_growth","electricity_growth",
    "distance_km","ice_eff_l_per_100km",
    "petrol_price_per_l","electricity_price_per_kwh",
    "predicted_savings","rule_ev_cost","rule_ice_cost",
]

buffer, written = [], 0
for row in scenario_rows():
    buffer.append(row)
    if len(buffer) >= BATCH_ROWS:
        pd.DataFrame(buffer, columns=cols).to_csv(
            OUTPUT_CSV, mode="a", index=False, header=(written == 0)
        )
        written += len(buffer)
        print(f"Wrote {written:,} rows...")
        buffer.clear()

if buffer:
    pd.DataFrame(buffer, columns=cols).to_csv(
        OUTPUT_CSV, mode="a", index=False, header=(written == 0)
    )
    written += len(buffer)

print(f"Done. Total rows written: {written:,}")
print(f"Saved CSV: {Path(OUTPUT_CSV).resolve()}")
  