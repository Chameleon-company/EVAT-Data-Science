"""
schemas.py
----------
Pydantic request and response models for the FastAPI application.
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class EVInput(BaseModel):
    """
    Specify the EV either by make/model lookup OR by providing raw consumption.
    At least one of (make + model) or consumption_kwh_per_100km must be given.
    """
    make: Optional[str] = Field(None, example="Tesla", description="EV manufacturer name")
    model: Optional[str] = Field(None, example="Model 3", description="EV model name")
    year: Optional[int] = Field(None, ge=2010, le=2030, example=2024)
    variant: Optional[str] = Field(None, example="Standard Range RWD")

    # Override — use when the vehicle is not in the database
    consumption_kwh_per_100km: Optional[float] = Field(
        None, gt=0, le=80, example=14.9,
        description="WLTP AC consumption in kWh/100 km. Overrides database lookup."
    )
    battery_kwh: Optional[float] = Field(
        None, gt=0, le=250, example=57.5,
        description="Usable battery capacity in kWh. Required if using consumption_kwh_per_100km."
    )

    @model_validator(mode="after")
    def check_ev_input(self) -> "EVInput":
        has_lookup = self.make and self.model
        has_manual = self.consumption_kwh_per_100km is not None
        if not has_lookup and not has_manual:
            raise ValueError(
                "Provide either (make + model) for a database lookup, "
                "or consumption_kwh_per_100km for a manual entry."
            )
        if has_manual and self.battery_kwh is None:
            raise ValueError(
                "battery_kwh is required when providing consumption_kwh_per_100km manually."
            )
        return self


class ICEInput(BaseModel):
    """
    Specify the ICE vehicle either by make/model lookup OR by raw consumption.
    """
    make: Optional[str] = Field(None, example="Toyota", description="ICE vehicle manufacturer")
    model: Optional[str] = Field(None, example="RAV4", description="ICE vehicle model")
    year: Optional[int] = Field(None, ge=1990, le=2030, example=2024)
    variant: Optional[str] = Field(None, example="GX Petrol AWD")

    consumption_l_per_100km: Optional[float] = Field(
        None, gt=0, le=50, example=7.2,
        description="WLTP combined consumption in L/100 km. Overrides database lookup."
    )
    fuel_type: Optional[str] = Field(
        None, example="Petrol",
        description="Required when providing consumption_l_per_100km. "
                    "One of: Petrol, Petrol95, Petrol98, E10, Diesel, LPG"
    )

    @model_validator(mode="after")
    def check_ice_input(self) -> "ICEInput":
        has_lookup = self.make and self.model
        has_manual = self.consumption_l_per_100km is not None
        if not has_lookup and not has_manual:
            raise ValueError(
                "Provide either (make + model) for a database lookup, "
                "or consumption_l_per_100km + fuel_type for a manual entry."
            )
        if has_manual and not self.fuel_type:
            raise ValueError("fuel_type is required when providing consumption_l_per_100km.")
        return self


class UsageParams(BaseModel):
    """Optional driving behaviour and usage parameters."""
    annual_km: float = Field(
        13_100, gt=0, le=200_000,
        description="Expected annual driving distance in km. Default: ABS 2020 Australian average."
    )
    driving_style: Literal["city", "mixed", "highway"] = Field(
        "mixed",
        description="Typical driving pattern. Affects real-world EV efficiency adjustment."
    )
    vehicle_age_years: float = Field(
        0.0, ge=0, le=20,
        description="Current age of the EV in years. Older vehicles have degraded batteries."
    )
    avg_temp_celsius: Optional[float] = Field(
        None, ge=-30, le=55,
        description="Average ambient temperature (°C). If None, state capital average is used."
    )
    vehicle_lifetime_years: float = Field(
        15.0, gt=0, le=30,
        description="Expected total ownership period for lifetime calculations."
    )


class PredictRequest(BaseModel):
    ev: EVInput
    ice: ICEInput
    state: Literal["NSW", "ACT", "VIC", "QLD", "SA", "WA", "TAS", "NT"] = Field(
        ...,
        description="Australian state/territory. Determines grid emission intensity."
    )
    usage: UsageParams = Field(default_factory=UsageParams)
    include_lifecycle: bool = Field(
        True,
        description="Include battery manufacturing emissions in EV total (well-to-wheel)."
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class EVEmissions(BaseModel):
    wltp_consumption_kwh_per_100km: float
    real_world_consumption_kwh_per_100km: float
    grid_intensity_g_per_kwh: float
    operational_g_per_km: float
    manufacturing_g_per_km: float
    total_g_per_km: float


class ICEEmissions(BaseModel):
    consumption_l_per_100km: float
    fuel_type: str
    emission_factor_g_per_l: float
    operational_g_per_km: float


class Savings(BaseModel):
    co2_savings_g_per_km: float
    co2_savings_kg_per_year: float
    co2_savings_tonnes_lifetime: float
    percentage_reduction: float


class Financial(BaseModel):
    ev_cost_aud_per_km: float
    ice_cost_aud_per_km: float
    cost_saving_aud_per_km: float
    cost_saving_aud_per_year: float
    cost_saving_aud_lifetime: float


class Equivalents(BaseModel):
    trees_planted_equivalent_per_year: int
    petrol_litres_saved_per_year: float


class PredictResponse(BaseModel):
    ev: EVEmissions
    ice: ICEEmissions
    savings: Savings
    financial: Financial
    equivalents: Equivalents
    state: str
    driving_style: str
    real_world_adjustment_factor: float
    include_lifecycle: bool
    ev_make: Optional[str] = None
    ev_model: Optional[str] = None
    ice_make: Optional[str] = None
    ice_model: Optional[str] = None


class VehicleListResponse(BaseModel):
    makes: list[str]


class ModelListResponse(BaseModel):
    make: str
    models: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
