from pydantic import BaseModel, confloat
from typing import Literal

class CostRequest(BaseModel):
    distance_km: confloat(gt=0)
    electricity_price_per_kwh: confloat(gt=0)
    ice_eff_l_per_100km: confloat(gt=0)
    petrol_price_per_l: confloat(gt=0)

class CostResponse(BaseModel):
    predicted_savings: float
    currency: Literal["AUD"] = "AUD"
    model_version: str = "gbm_v1"