import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Load the bundle (model + metadata)
with open("kproto_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

kproto = bundle["model"]
FEATURE_COLS = bundle["feature_cols"]
CAT_COLS = bundle["cat_cols"]  # indices relative to FEATURE_COLS

app = Flask(__name__)

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure numeric fields are numeric and category fields are strings.
    Missing fields are filled with empty strings for categoricals and 0 for numerics.
    Adjust this mapping if your training schema changes.
    """
    numeric_cols = ["weekly_km", "fuel_efficiency", "monthly_fuel_spend"]
    categorical_cols = [c for c in FEATURE_COLS if c not in numeric_cols]

    # Ensure columns exist; add missing as NaN/empty
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Type coercion
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in categorical_cols:
        # Keep exact strings used in training (e.g., "Yes", "No", "Home", etc.)
        df[col] = df[col].fillna("").astype(str)

    # Reorder columns to match training
    df = df[FEATURE_COLS]
    return df

@app.route("/healthz", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expect JSON body containing at least the fields in FEATURE_COLS.
    Extra fields are ignored.
    Returns: {"cluster": <int>}
    """
    try:
        payload = request.get_json(force=True, silent=False)

        # Accept single object or list of objects; standardise to list
        if isinstance(payload, dict):
            records = [payload]
        elif isinstance(payload, list):
            records = payload
        else:
            return jsonify({"error": "Invalid JSON payload"}), 400

        df = pd.DataFrame(records)
        df = coerce_types(df)

        # Convert to numpy with mixed types; kmodes handles categoricals as strings
        X = df.to_numpy()

        # Predict
        clusters = kproto.predict(X, categorical=CAT_COLS)
        # Return single prediction if input was a single object
        if len(records) == 1:
            return jsonify({"cluster": int(clusters[0])}), 200
        else:
            return jsonify({"clusters": [int(c) for c in clusters]}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Flask dev server (Render will run via gunicorn)
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
