from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from routing_engine.engine import plan_route, load_dataset, build_nodes

import numpy as np
import pandas as pd

app = FastAPI(title="EVAT Routing API")
DATA_PATH = "data/combined_data.csv"
_df = load_dataset(DATA_PATH)
_nodes = build_nodes(_df)

class RouteRequest(BaseModel):
    start_idx: int = 0
    goal_idx: int = 10
    ev_range_km: float = 35.0
    k_neighbors: int = 6
    assumed_speed_kmh: float = 60.0
    alpha_weather: float = 0.15
    beta_traffic: float = 0.10
    charge_penalty_min: float = 15.0
    mode: str = "FALLBACK"

def to_native(x):
    # Convert anything numpy/pandas into plain Python JSON-safe values
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        if not np.isfinite(v):  # inf/NaN -> None
            return None
        return v
    if isinstance(x, (np.ndarray,)):
        return [to_native(v) for v in x.tolist()]
    if isinstance(x, pd.Series):
        return [to_native(v) for v in x.to_list()]
    if isinstance(x, pd.DataFrame):
        return [to_native(r) for r in x.to_dict(orient="records")]
    if isinstance(x, dict):
        return {k: to_native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_native(v) for v in x]
    return x  # plain python (int/float/str/bool/None)

@app.get("/health")
def health():
    return {"status": "ok", "nodes": int(len(_nodes))}

@app.get("/stations")
def stations():
    return to_native(_nodes)

@app.post("/route")
def route(req: RouteRequest):
    try:
        result = plan_route(
            data_path=DATA_PATH,
            start_idx=req.start_idx, goal_idx=req.goal_idx,
            EV_RANGE_KM=req.ev_range_km, K_NEIGHBORS=req.k_neighbors,
            ASSUMED_SPEED_KMH=req.assumed_speed_kmh,
            ALPHA_WEATHER=req.alpha_weather, BETA_TRAFFIC=req.beta_traffic,
            CHARGE_TIME_MIN=req.charge_penalty_min, MODE=req.mode
        )
        return JSONResponse(content=to_native(result))
    except Exception as e:
        return JSONResponse(status_code=500, content={"status":"error","detail":str(e)})
