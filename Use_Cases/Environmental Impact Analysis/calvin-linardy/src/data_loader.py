"""
data_loader.py
--------------
Utilities for loading and validating the vehicle and emission-factor datasets.
All paths are relative so the module works regardless of where the repo is cloned.
"""

from pathlib import Path
import pandas as pd

# Root of the calvin-linardy package (two levels up from this file: src/ → root)
_PKG_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PKG_ROOT / "data"


def _load_csv(filename: str) -> pd.DataFrame:
    path = _DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run the project from its root directory or check SETUP_GUIDE.md."
        )
    return pd.read_csv(path)


def load_ev_vehicles() -> pd.DataFrame:
    """
    Load the EV vehicle database.

    Key columns
    -----------
    Make, Model, Year, Variant, BodyStyle, Segment,
    Battery_kWh, Range_km_WLTP, Consumption_kWh_per_100km
    """
    df = _load_csv("ev_vehicles.csv")
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
    df = _load_csv("ice_vehicles.csv")
    df["Consumption_Combined_L100km"] = pd.to_numeric(
        df["Consumption_Combined_L100km"], errors="coerce"
    )
    df = df.dropna(subset=["Consumption_Combined_L100km"])
    return df.reset_index(drop=True)


def load_grid_factors() -> pd.DataFrame:
    """Return state-level grid emission factors as a DataFrame."""
    return _load_csv("grid_emission_factors.csv")


def load_fuel_factors() -> pd.DataFrame:
    """Return fuel-type emission factors as a DataFrame."""
    return _load_csv("fuel_emission_factors.csv")


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
