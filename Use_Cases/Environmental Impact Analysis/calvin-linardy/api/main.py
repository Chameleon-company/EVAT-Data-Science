"""
main.py
-------
FastAPI application for the Environmental Impact Analysis use case.

Run locally:
    uvicorn api.main:app --reload --port 8001

Endpoints
---------
POST /api/environmental-impact/predict       Full CO2 comparison
GET  /api/environmental-impact/vehicles/ev   List EV makes
GET  /api/environmental-impact/vehicles/ev/{make}/models
GET  /api/environmental-impact/vehicles/ice  List ICE makes
GET  /api/environmental-impact/vehicles/ice/{make}/models
GET  /api/environmental-impact/health        Health check
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    PredictRequest, PredictResponse,
    EVEmissions, ICEEmissions, Savings, Financial, Equivalents,
    VehicleListResponse, ModelListResponse, HealthResponse,
)
from src import calculator, model as xgb_model, data_loader
from src.config import (
    GRID_EMISSION_FACTORS,
    DEFAULT_ANNUAL_KM,
)

# State capital average temperatures (°C) used when avg_temp_celsius is not provided
_STATE_AVG_TEMPS = {
    "NSW": 17.7,   # Sydney annual average
    "ACT": 13.3,   # Canberra annual average
    "VIC": 14.9,   # Melbourne annual average
    "QLD": 25.0,   # Brisbane annual average
    "SA":  17.3,   # Adelaide annual average
    "WA":  18.7,   # Perth annual average
    "TAS": 12.4,   # Hobart annual average
    "NT":  28.4,   # Darwin annual average
}

app = FastAPI(
    title="EVAT Environmental Impact Analysis API",
    description=(
        "Calculates CO2 savings when replacing an ICE vehicle with an EV. "
        "Uses Australian state-specific grid emission factors (DCCEEW 2023) "
        "and an XGBoost model to adjust for real-world driving conditions."
    ),
    version="2.0.0",
    docs_url="/api/environmental-impact/docs",
    redoc_url="/api/environmental-impact/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the XGBoost model once at startup (avoids disk I/O on every request)
_model_bundle: dict | None = None


@app.on_event("startup")
def load_model_on_startup():
    global _model_bundle
    try:
        _model_bundle = xgb_model.load_model()
        print("XGBoost real-world adjustment model loaded.")
    except FileNotFoundError:
        print(
            "WARNING: XGBoost model not found. "
            "Run `python train.py` to train it. "
            "Predictions will use WLTP figures without real-world adjustment."
        )
        _model_bundle = None


# ---------------------------------------------------------------------------
# Helper: resolve EV inputs (lookup or manual)
# ---------------------------------------------------------------------------

def _resolve_ev(ev_input, state: str):
    """Return (consumption_kwh_per_100km, battery_kwh, make, model)."""
    if ev_input.consumption_kwh_per_100km is not None:
        return (
            ev_input.consumption_kwh_per_100km,
            ev_input.battery_kwh,
            ev_input.make,
            ev_input.model,
        )

    row = data_loader.get_ev_vehicle(
        ev_input.make, ev_input.model,
        year=ev_input.year, variant=ev_input.variant,
    )
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"EV '{ev_input.make} {ev_input.model}' not found in database. "
                   "Provide consumption_kwh_per_100km and battery_kwh instead.",
        )
    return (
        float(row["Consumption_kWh_per_100km"]),
        float(row["Battery_kWh"]),
        row["Make"],
        row["Model"],
    )


def _resolve_ice(ice_input):
    """Return (consumption_l_per_100km, fuel_type, make, model)."""
    if ice_input.consumption_l_per_100km is not None:
        return (
            ice_input.consumption_l_per_100km,
            ice_input.fuel_type,
            ice_input.make,
            ice_input.model,
        )

    row = data_loader.get_ice_vehicle(
        ice_input.make, ice_input.model,
        year=ice_input.year, variant=ice_input.variant,
    )
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"ICE vehicle '{ice_input.make} {ice_input.model}' not found in database. "
                   "Provide consumption_l_per_100km and fuel_type instead.",
        )
    return (
        float(row["Consumption_Combined_L100km"]),
        row["FuelType"],
        row["Make"],
        row["Model"],
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post(
    "/api/environmental-impact/predict",
    response_model=PredictResponse,
    summary="Compare EV vs ICE CO2 emissions",
    tags=["Prediction"],
)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Calculate CO2 savings when replacing an ICE vehicle with an EV.

    **Two modes for specifying vehicles:**
    1. **Database lookup** — provide `make` and `model` (and optionally `year`/`variant`).
    2. **Manual entry** — provide raw `consumption_kwh_per_100km` + `battery_kwh` for EV,
       or `consumption_l_per_100km` + `fuel_type` for ICE.

    **Key output fields:**
    - `savings.co2_savings_g_per_km` — grams of CO2 saved per km driven
    - `savings.co2_savings_tonnes_lifetime` — total tonnes over vehicle lifetime
    - `savings.percentage_reduction` — relative reduction vs ICE
    - `financial.cost_saving_aud_per_year` — annual running cost saving (AUD)
    """
    # --- Resolve vehicle specs ---
    ev_kwh, ev_battery, ev_make, ev_model_name = _resolve_ev(request.ev, request.state)
    ice_l, ice_fuel, ice_make, ice_model_name = _resolve_ice(request.ice)

    usage = request.usage
    avg_temp = usage.avg_temp_celsius
    if avg_temp is None:
        avg_temp = _STATE_AVG_TEMPS.get(request.state, 20.0)

    # --- Real-world adjustment ---
    if _model_bundle is not None:
        try:
            rw_factor = xgb_model.predict_adjustment(
                avg_temp_celsius=avg_temp,
                driving_style=usage.driving_style,
                vehicle_age_years=usage.vehicle_age_years,
                battery_capacity_kwh=ev_battery,
                model_bundle=_model_bundle,
            )
        except Exception:
            rw_factor = 1.0   # Graceful fallback
    else:
        rw_factor = 1.0

    # --- Core calculation ---
    result = calculator.calculate(
        ev_consumption_kwh_per_100km=ev_kwh,
        ev_battery_kwh=ev_battery,
        ice_consumption_l_per_100km=ice_l,
        ice_fuel_type=ice_fuel,
        state=request.state,
        real_world_adjustment=rw_factor,
        annual_km=usage.annual_km,
        vehicle_lifetime_years=usage.vehicle_lifetime_years,
        driving_style=usage.driving_style,
        include_lifecycle=request.include_lifecycle,
    )

    return PredictResponse(
        ev=EVEmissions(
            wltp_consumption_kwh_per_100km=result.ev_wltp_consumption_kwh_per_100km,
            real_world_consumption_kwh_per_100km=result.ev_real_world_consumption_kwh_per_100km,
            grid_intensity_g_per_kwh=result.ev_grid_intensity_g_per_kwh,
            operational_g_per_km=result.ev_operational_g_per_km,
            manufacturing_g_per_km=result.ev_manufacturing_g_per_km,
            total_g_per_km=result.ev_total_g_per_km,
        ),
        ice=ICEEmissions(
            consumption_l_per_100km=result.ice_consumption_l_per_100km,
            fuel_type=result.ice_fuel_type,
            emission_factor_g_per_l=result.ice_emission_factor_g_per_l,
            operational_g_per_km=result.ice_operational_g_per_km,
        ),
        savings=Savings(
            co2_savings_g_per_km=result.co2_savings_g_per_km,
            co2_savings_kg_per_year=result.co2_savings_kg_per_year,
            co2_savings_tonnes_lifetime=result.co2_savings_tonnes_lifetime,
            percentage_reduction=result.percentage_reduction,
        ),
        financial=Financial(
            ev_cost_aud_per_km=result.ev_cost_aud_per_km,
            ice_cost_aud_per_km=result.ice_cost_aud_per_km,
            cost_saving_aud_per_km=result.cost_saving_aud_per_km,
            cost_saving_aud_per_year=result.cost_saving_aud_per_year,
            cost_saving_aud_lifetime=result.cost_saving_aud_lifetime,
        ),
        equivalents=Equivalents(
            trees_planted_equivalent_per_year=result.trees_planted_equivalent_per_year,
            petrol_litres_saved_per_year=result.petrol_litres_saved_per_year,
        ),
        state=result.state,
        driving_style=result.driving_style,
        real_world_adjustment_factor=result.real_world_adjustment_factor,
        include_lifecycle=result.include_lifecycle,
        ev_make=ev_make,
        ev_model=ev_model_name,
        ice_make=ice_make,
        ice_model=ice_model_name,
    )


@app.get(
    "/api/environmental-impact/vehicles/ev",
    response_model=VehicleListResponse,
    summary="List available EV makes",
    tags=["Vehicle Lookup"],
)
def list_ev_makes():
    return VehicleListResponse(makes=data_loader.list_ev_makes())


@app.get(
    "/api/environmental-impact/vehicles/ev/{make}/models",
    response_model=ModelListResponse,
    summary="List EV models for a given make",
    tags=["Vehicle Lookup"],
)
def list_ev_models(make: str):
    models = data_loader.list_ev_models(make)
    if not models:
        raise HTTPException(status_code=404, detail=f"No EV models found for make '{make}'.")
    return ModelListResponse(make=make, models=models)


@app.get(
    "/api/environmental-impact/vehicles/ice",
    response_model=VehicleListResponse,
    summary="List available ICE vehicle makes",
    tags=["Vehicle Lookup"],
)
def list_ice_makes():
    return VehicleListResponse(makes=data_loader.list_ice_makes())


@app.get(
    "/api/environmental-impact/vehicles/ice/{make}/models",
    response_model=ModelListResponse,
    summary="List ICE models for a given make",
    tags=["Vehicle Lookup"],
)
def list_ice_models(make: str):
    models = data_loader.list_ice_models(make)
    if not models:
        raise HTTPException(status_code=404, detail=f"No ICE models found for make '{make}'.")
    return ModelListResponse(make=make, models=models)


@app.get(
    "/api/environmental-impact/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Utility"],
)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_model_bundle is not None,
        version="2.0.0",
    )
