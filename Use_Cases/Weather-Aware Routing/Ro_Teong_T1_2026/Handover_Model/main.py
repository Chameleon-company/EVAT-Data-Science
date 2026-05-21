from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services import get_route, get_weather, get_charging_stations
from model import predict_trip, needs_charging, traffic_energy_factor, traffic_condition_label
from config import DEFAULT_SOC_PCT

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TripRequest(BaseModel):
    origin: str
    destination: str
    ac_on: bool = True

@app.post("/predict")
def predict(req: TripRequest):
    route = get_route(req.origin, req.destination)
    if not route:
        raise HTTPException(status_code=404, detail="No route found. Check both addresses.")

    leg, elevations, polyline = route
    start = (leg["start_location"]["lat"], leg["start_location"]["lng"])
    end   = (leg["end_location"]["lat"], leg["end_location"]["lng"])
    weather = get_weather(start[0], start[1])

    # pass ac_on through to physics model (bug 1 fix)
    result = predict_trip(leg["steps"], elevations, weather, ac_on=req.ac_on)

    # real-time traffic durations
    duration_normal_s  = leg["duration"]["value"]
    duration_traffic_s = leg.get("duration_in_traffic", {}).get("value", duration_normal_s)

    # apply traffic factor only to adjusted energy and SOC — keep nominal clean (bug 2 fix)
    factor = traffic_energy_factor(duration_normal_s, duration_traffic_s)
    result["energy_with_ac_kwh"]       = round(result["energy_with_ac_kwh"] * factor, 3)
    result["soc_needed_pct"]           = round(result["soc_needed_pct"] * factor, 1)
    result["soc_with_contingency_pct"] = round(result["soc_with_contingency_pct"] * factor, 1)

    charging_required = needs_charging(result["soc_with_contingency_pct"], DEFAULT_SOC_PCT)

    charging_stops = []
    if charging_required:
        charging_stops = get_charging_stations(end[0], end[1], radius_m=2000)

    return {
        "origin_resolved": leg["start_address"],
        "destination_resolved": leg["end_address"],
        "origin_coords": leg["start_location"],
        "destination_coords": leg["end_location"],
        "distance_km": round(leg["distance"]["value"] / 1000, 1),
        "duration_min": round(duration_normal_s / 60, 0),
        "duration_in_traffic_min": round(duration_traffic_s / 60, 0),
        "traffic_condition": traffic_condition_label(factor),
        "polyline": polyline,
        "steps": [
            {
                "instruction": s["html_instructions"],
                "distance_m": s["distance"]["value"],
                "duration_s": s["duration"]["value"],
                "start_location": s["start_location"],
                "end_location": s["end_location"],
            }
            for s in leg["steps"]
        ],
        **result,
        "weather": weather,
        "charging_required": charging_required,
        "charging_stops": charging_stops,
    }