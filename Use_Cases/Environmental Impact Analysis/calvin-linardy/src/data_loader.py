"""
data_loader.py
--------------
Utilities for loading and validating the vehicle and emission-factor datasets.

Data source: MongoDB Atlas (EVAT database) when MONGO_URI env var is set.
Falls back to local CSV files when MONGO_URI is not set, so train.py and
local development work without a database connection.

MongoDB collections used:
  environmental_impact_ev_vehicles
  environmental_impact_ice_vehicles
  grid_emission_factors
  fuel_emission_factors
"""

import os
from pathlib import Path

import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PKG_ROOT / "data"
_MONGO_URI = os.getenv("MONGO_URI")
_DB_NAME = "EVAT"

_mongo_client = None


def _get_db():
    global _mongo_client
    if _mongo_client is None:
        from pymongo import MongoClient
        _mongo_client = MongoClient(_MONGO_URI)
    return _mongo_client[_DB_NAME]


def _load_from_mongo(collection: str) -> pd.DataFrame:
    db = _get_db()
    docs = list(db[collection].find({}, {"_id": 0}))
    return pd.DataFrame(docs)


def _load_csv(filename: str) -> pd.DataFrame:
    path = _DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Set MONGO_URI env var or check SETUP_GUIDE.md."
        )
    return pd.read_csv(path)


def _load(mongo_collection: str, csv_filename: str) -> pd.DataFrame:
    if _MONGO_URI:
        return _load_from_mongo(mongo_collection)
    return _load_csv(csv_filename)


def load_ev_vehicles() -> pd.DataFrame:
    """
    Load the EV vehicle database.

    Key columns
    -----------
    Make, Model, Year, Variant, BodyStyle, Segment,
    Battery_kWh, Range_km_WLTP, Consumption_kWh_per_100km
    """
    df = _load("environmental_impact_ev_vehicles", "ev_vehicles.csv")
    df["Consumption_kWh_per_100km"] = pd.to_numeric(
        df["Consumption_kWh_per_100km"], errors="coerce"
    )
    df["Battery_kWh"] = pd.to_numeric(df["Battery_kWh"], errors="coerce")
    df = df.dropna(subset=["Consumption_kWh_per_100km", "Battery_kWh"])
    return df.reset_index(drop=True)


def load_ice_vehicles() -> pd.DataFrame:
    """
    Load the ICE vehicle database.

    Key columns
    -----------
    Make, Model, Year, Variant, FuelType, BodyStyle, Segment,
    Consumption_Combined_L100km, CO2_Combined_gkm
    """
    df = _load("environmental_impact_ice_vehicles", "ice_vehicles.csv")
    df["Consumption_Combined_L100km"] = pd.to_numeric(
        df["Consumption_Combined_L100km"], errors="coerce"
    )
    df = df.dropna(subset=["Consumption_Combined_L100km"])
    return df.reset_index(drop=True)


def load_grid_factors() -> pd.DataFrame:
    """Return state-level grid emission factors as a DataFrame."""
    return _load("grid_emission_factors", "grid_emission_factors.csv")


def load_fuel_factors() -> pd.DataFrame:
    """Return fuel-type emission factors as a DataFrame."""
    return _load("fuel_emission_factors", "fuel_emission_factors.csv")


def get_ev_vehicle(make: str, model: str, year: int = None, variant: str = None) -> pd.Series | None:
    """
    Look up a specific EV. Returns the best-matching row or None.

    Matching priority
    -----------------
    1. make + model + year + variant  (exact)
    2. make + model + year            (first matching variant)
    3. make + model                   (most recent year)
    """
    df = load_ev_vehicles()

    mask = (df["Make"].str.lower() == make.lower()) & \
           (df["Model"].str.lower() == model.lower())
    subset = df[mask]
    if subset.empty:
        return None

    if year is not None:
        y_sub = subset[subset["Year"] == year]
        if not y_sub.empty:
            subset = y_sub

    if variant is not None:
        v_sub = subset[subset["Variant"].str.lower() == variant.lower()]
        if not v_sub.empty:
            subset = v_sub

    # Return the most recent year entry
    return subset.sort_values("Year", ascending=False).iloc[0]


def get_ice_vehicle(make: str, model: str, year: int = None, variant: str = None) -> pd.Series | None:
    """Look up a specific ICE vehicle. Matching logic mirrors get_ev_vehicle."""
    df = load_ice_vehicles()

    mask = (df["Make"].str.lower() == make.lower()) & \
           (df["Model"].str.lower() == model.lower())
    subset = df[mask]
    if subset.empty:
        return None

    if year is not None:
        y_sub = subset[subset["Year"] == year]
        if not y_sub.empty:
            subset = y_sub

    if variant is not None:
        v_sub = subset[subset["Variant"].str.lower() == variant.lower()]
        if not v_sub.empty:
            subset = v_sub

    return subset.sort_values("Year", ascending=False).iloc[0]


def list_ev_makes() -> list[str]:
    return sorted(load_ev_vehicles()["Make"].unique().tolist())


def list_ice_makes() -> list[str]:
    return sorted(load_ice_vehicles()["Make"].unique().tolist())


def list_ev_models(make: str) -> list[str]:
    df = load_ev_vehicles()
    return sorted(df[df["Make"].str.lower() == make.lower()]["Model"].unique().tolist())


def list_ice_models(make: str) -> list[str]:
    df = load_ice_vehicles()
    return sorted(df[df["Make"].str.lower() == make.lower()]["Model"].unique().tolist())
