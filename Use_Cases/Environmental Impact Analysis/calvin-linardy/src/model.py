"""
model.py
--------
CatBoost model that predicts a real-world efficiency adjustment factor for EVs.

Why this exists
---------------
The WLTP test cycle is conducted under controlled laboratory conditions: 23 °C
ambient, a fixed speed profile, no HVAC load, and a new battery. Real-world
consumption deviates from WLTP depending on:
  - Ambient temperature (cold weather increases battery resistance and HVAC use)
  - Driving style (highway at 110+ km/h is significantly less efficient for EVs)
  - Battery age (capacity and internal resistance degrade ~2% per year)
  - Battery size (smaller packs have less thermal headroom — higher real-world factor)
  - Auxiliary loads (headlights, seat heating, fast charging heat cycles)

The CatBoost model predicts a multiplier (real_world_adjustment_factor) that is
applied to the WLTP figure before the CO2 calculation. A value of 1.10 means
the vehicle will consume 10% more energy than WLTP in these conditions.

Model selection: CatBoost was chosen after comparing Ridge, Random Forest,
GradientBoosting, XGBoost, LightGBM, and CatBoost on the same dataset.
CatBoost achieved the best CV R² (0.9545) and lowest MAE (0.02076).

Training data
-------------
Generated synthetically from published physics relationships, not from measured
vehicle telemetry. References:
  - ICCT 2022 "Real-world EV range"
  - RAC Foundation 2021 EV temperature study
  - Transport & Environment 2023 EV winter efficiency report

When real fleet telemetry data becomes available, the generate_training_data()
function can be replaced and the model retrained — the API interface remains
unchanged.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import RW_ADJUSTMENT_MIN, RW_ADJUSTMENT_MAX

_PKG_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = _PKG_ROOT / "models"
_MODEL_PATH = _MODEL_DIR / "rw_adjustment_catboost.pkl"

# ---------------------------------------------------------------------------
# Synthetic training data generator
# ---------------------------------------------------------------------------

def generate_training_data(n_samples: int = 8_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic training data for the real-world adjustment model.

    Each sample represents one EV driver in one set of conditions.
    The target (real_world_factor) is derived from physics-based rules with
    added Gaussian noise to simulate real-world variance.

    Features
    --------
    avg_temp_celsius       Annual average ambient temperature
    driving_style          0 = city, 1 = mixed, 2 = highway
    vehicle_age_years      Age of the EV at time of measurement
    battery_capacity_kwh   Usable battery pack size
    climate_zone           0 = temperate, 1 = tropical/arid, 2 = cold

    Target
    ------
    real_world_factor      Ratio of actual to WLTP consumption
                           e.g. 1.12 means 12% worse than WLTP
    """
    rng = np.random.default_rng(seed)

    # --- Feature distributions based on Australian climate zones ---
    avg_temp = rng.uniform(-5.0, 42.0, n_samples)   # °C; full AU range
    driving_style = rng.choice([0, 1, 2], n_samples, p=[0.25, 0.50, 0.25])
    vehicle_age_years = rng.uniform(0.0, 12.0, n_samples)
    battery_capacity_kwh = rng.uniform(30.0, 120.0, n_samples)

    # Climate zone: derived from temperature
    climate_zone = np.where(avg_temp < 10, 2,         # cold
                   np.where(avg_temp > 28, 1, 0))     # tropical/arid / temperate

    # --- Physics-based target construction ---

    # Base: new EV, mixed driving, 20 °C → factor ≈ 1.05 (EVs typically 5% worse
    # than WLTP in real world due to auxiliary loads and real traffic patterns)
    base = 1.05

    # Temperature effect
    # Below 15 °C: battery resistance rises, HVAC heating draws power.
    # Above 30 °C: HVAC cooling draws power.
    temp_effect = np.where(
        avg_temp < 15,
        0.008 * (15.0 - avg_temp),      # +0.8% per °C below 15 — up to +16% at -5 °C
        np.where(avg_temp > 30,
                 0.003 * (avg_temp - 30.0),  # +0.3% per °C above 30
                 0.0)
    )

    # Driving style effect
    # City: regenerative braking partially compensates — slight improvement vs mixed.
    # Highway at high speed: aerodynamic drag dominates, EVs lose WLTP advantage.
    style_effect = np.where(driving_style == 0, -0.04,     # city: -4%
                   np.where(driving_style == 2,  0.12,     # highway: +12%
                                                 0.03))    # mixed: +3%

    # Battery degradation: ~2% increase in consumption per year (conservative)
    # Larger batteries degrade slower (better thermal management)
    age_coeff = 0.020 - 0.0002 * (battery_capacity_kwh - 60) / 30
    age_coeff = np.clip(age_coeff, 0.010, 0.030)
    age_effect = age_coeff * vehicle_age_years

    # Battery size direct efficiency effect
    # Smaller packs (<50 kWh) run closer to their limits — less headroom means
    # less ability to buffer temperature and load spikes, raising real-world factor.
    # Larger packs (>90 kWh) have more thermal headroom — slightly better real-world.
    battery_size_effect = -0.0015 * (battery_capacity_kwh - 60)  # ±0.045 across 30–120 kWh range
    battery_size_effect = np.clip(battery_size_effect, -0.045, 0.045)

    # Battery size moderates cold-weather impact
    # Larger packs have better thermal management hardware — 3x stronger than before
    battery_temp_mod = -0.0025 * (battery_capacity_kwh - 60) * np.abs(avg_temp - 20) / 20
    battery_temp_mod = np.clip(battery_temp_mod, -0.08, 0.08)

    # Gaussian noise (real-world variance from driving behaviour, road type, etc.)
    noise = rng.normal(0, 0.025, n_samples)

    real_world_factor = base + temp_effect + style_effect + age_effect + battery_size_effect + battery_temp_mod + noise
    real_world_factor = np.clip(real_world_factor, RW_ADJUSTMENT_MIN, RW_ADJUSTMENT_MAX)

    return pd.DataFrame({
        "avg_temp_celsius": avg_temp.round(1),
        "driving_style": driving_style,
        "vehicle_age_years": vehicle_age_years.round(1),
        "battery_capacity_kwh": battery_capacity_kwh.round(1),
        "climate_zone": climate_zone,
        "real_world_factor": real_world_factor.round(4),
    })


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(n_samples: int = 8_000, seed: int = 42) -> dict:
    """
    Generate synthetic data, train CatBoost, save model, and return metrics.

    The model is saved to models/rw_adjustment_catboost.pkl.

    Returns
    -------
    dict with keys: mae, rmse, r2, cv_r2_mean, cv_r2_std, n_train, n_test
    """
    _MODEL_DIR.mkdir(exist_ok=True)

    print("Generating synthetic training data...")
    df = generate_training_data(n_samples=n_samples, seed=seed)
    print(f"  {len(df):,} samples generated.")
    print(f"  Target stats — mean: {df['real_world_factor'].mean():.4f}, "
          f"std: {df['real_world_factor'].std():.4f}, "
          f"range: [{df['real_world_factor'].min():.3f}, {df['real_world_factor'].max():.3f}]")

    feature_cols = [
        "avg_temp_celsius",
        "driving_style",
        "vehicle_age_years",
        "battery_capacity_kwh",
        "climate_zone",
    ]
    X = df[feature_cols].values
    y = df["real_world_factor"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )

    print("\nTraining CatBoost model...")
    model = CatBoostRegressor(
        iterations=500,
        depth=4,
        learning_rate=0.08,
        l2_leaf_reg=1.0,
        random_seed=seed,
        verbose=0,
    )
    model.fit(X_train, y_train)

    # --- Hold-out evaluation ---
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2   = r2_score(y_test, y_pred)

    # --- 5-fold cross-validation on full dataset ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)

    print(f"\n  Hold-out   MAE : {mae:.5f}")
    print(f"  Hold-out   RMSE: {rmse:.5f}")
    print(f"  Hold-out   R²  : {r2:.4f}")
    print(f"  CV R²  (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # --- Feature importances ---
    importances = dict(zip(feature_cols, model.get_feature_importance().round(4)))
    print(f"\n  Feature importances: {importances}")

    # --- Save ---
    joblib.dump({"model": model, "feature_cols": feature_cols}, _MODEL_PATH)
    print(f"\nModel saved to {_MODEL_PATH}")

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 4),
        "cv_r2_mean": round(cv_scores.mean(), 4),
        "cv_r2_std": round(cv_scores.std(), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importances": importances,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model() -> dict:
    """Load the trained model bundle from disk."""
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {_MODEL_PATH}.\n"
            "Run `python train.py` first to train and save the model."
        )
    return joblib.load(_MODEL_PATH)


def predict_adjustment(
    avg_temp_celsius: float,
    driving_style: str,
    vehicle_age_years: float,
    battery_capacity_kwh: float,
    model_bundle: dict | None = None,
) -> float:
    """
    Predict the real-world efficiency adjustment factor for an EV.

    Parameters
    ----------
    avg_temp_celsius : float
        Average ambient temperature (°C) for the region/season.
        Use annual average for a general estimate.
    driving_style : str
        'city', 'mixed', or 'highway'.
    vehicle_age_years : float
        Age of the vehicle in years (0 = brand new).
    battery_capacity_kwh : float
        Usable battery capacity in kWh.
    model_bundle : dict | None
        Pre-loaded model bundle. If None, loads from disk (slower — avoid
        in high-throughput API scenarios; pass a pre-loaded bundle instead).

    Returns
    -------
    float
        Adjustment factor. Multiply EV WLTP consumption by this value to get
        the estimated real-world consumption.
        Example: 1.12 → 12% worse than WLTP.
    """
    style_map = {"city": 0, "mixed": 1, "highway": 2}
    style_int = style_map.get(driving_style.lower().strip())
    if style_int is None:
        raise ValueError(
            f"Unknown driving_style '{driving_style}'. "
            "Valid values: 'city', 'mixed', 'highway'."
        )

    climate_zone = 2 if avg_temp_celsius < 10 else (1 if avg_temp_celsius > 28 else 0)

    features = np.array([[
        avg_temp_celsius,
        style_int,
        vehicle_age_years,
        battery_capacity_kwh,
        climate_zone,
    ]])

    if model_bundle is None:
        model_bundle = load_model()

    raw = model_bundle["model"].predict(features)[0]
    return float(np.clip(raw, RW_ADJUSTMENT_MIN, RW_ADJUSTMENT_MAX))
