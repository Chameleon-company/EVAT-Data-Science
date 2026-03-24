# Weather-Aware Routing - Handover Documentation

**Document Version:** 1.1  
**Last Updated:** March 24, 2026  
**Scope:** `Use_Cases/Weather-Aware Routing/`

---

## Table of Contents
1. [Primary Goal](#primary-goal)
2. [Model Used](#model-used)
3. [System Components](#system-components)
4. [Dependencies](#dependencies)
5. [External APIs](#external-apis)
6. [Data Source](#data-source)
7. [Data Requirements](#data-requirements)
8. [Feature Engineering](#feature-engineering)
9. [Model Performance](#model-performance)
10. [Expected Output (UI Map)](#expected-output-ui-map)
11. [Deployment Guide](#deployment-guide)
12. [Execution Flow](#execution-flow)
13. [Troubleshooting](#troubleshooting)
14. [API Timeout - How to Solve](#api-timeout---how-to-solve)
15. [Future Enhancements](#future-enhancements)
16. [Appendix A - Complete Feature List](#appendix-a---complete-feature-list)
17. [Appendix B - Sample API Responses](#appendix-b---sample-api-responses)

---

## Primary Goal

The weather-aware routing workstream aims to combine **traffic + weather + EV charging station data** to improve EV mobility decisions, with a target outcome of **dynamic weather-aware route planning** for safer and more efficient travel.  

- Uday track explicitly frames the objective as dynamic weather-aware routing for EV drivers.  
- Duy track focuses on prediction and geospatial feature extraction that can support routing UX.

---

## Model Used

### 1) Duy track (`Use_Cases/Weather-Aware Routing/Duy Pham/`)
- Production artifact loaded by API: `ev_model.pkl`.
- Verified model class: **`sklearn.ensemble.RandomForestRegressor`**.
- API feature input list in code:
  - `Year`
  - `SHAPE_Length`
  - `dist_to_nearest_ev_m`
  - `ev_within_500m`
  - `avg_temp`
  - `total_prcp`

### 2) Uday track (`Use_Cases/Weather-Aware Routing/Uday kiran reddy Neerudu/`)
- Routing runtime itself is **algorithmic pathfinding** (graph + weighted cost + Dijkstra), not a supervised model inference API.
- README reports Sprint 3 used a **Random Forest regression** model for weather sensitivity scoring.

---

## System Components

## A) Uday Routing Stack
1. **`engine sprint 4.py`**
   - Dataset loading + fallback scoring
   - Node construction
   - Edge construction (KNN + EV range)
   - Cost function (distance + weather + traffic)
   - Dijkstra with stop/charge penalty
2. **`main - sprint 4.py`**
   - FastAPI wrapper with `/health`, `/stations`, `/route`
3. **`adapters sprint 4.py`**
   - Adapter scaffold for weather/traffic (currently fallback/default behavior)
4. **`ui_app sprint 4.py`**
   - Streamlit UI to call API and draw route map
5. **`utils -sprint 4.py`**
   - Haversine distance and normalization helpers

## B) Duy Prediction + Web Stack
1. **`api.py`**
   - Flask API with `/` and `/predict`
   - Geospatial feature generation from input coordinates
   - Model inference via `ev_model.pkl`
2. **`apiClient.ts`**
   - Typed API client, validation, error handling
3. **`MapComponent.tsx`**
   - Google Maps click-to-select location
   - Calls `/predict` and renders prediction + route insights panel
4. **`App.tsx` / `src/index.tsx`**
   - App shell and entry point
5. **`useEVPrediction.ts`**
   - Optional reusable hook for prediction workflow

---

## Dependencies

## Python (backend)
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `flask`
- `flask-cors` (optional fallback exists if unavailable)
- `geopandas`
- `shapely`
- (Uday stack additionally uses `fastapi`, `pydantic`)

## Frontend (React)
- `react`
- `react-dom`
- `typescript`
- `react-scripts`
- `@react-google-maps/api`

---

## External APIs

1. **Google Maps JavaScript API**
   - Used by Duy frontend (`MapComponent.tsx`) for interactive map rendering.

2. **Weather / Traffic live APIs**
   - Not yet actively integrated in Uday adapter runtime; adapter currently returns defaults (scaffold for future live integration).

---

## Data Source

Current local files used by code paths include:

## Duy
- `Traffic data.csv`
- `EV stations data.csv`
- `weather_by_station_2023.csv`
- `ev_model.pkl`

## Uday
- Combined weather/traffic/EV datasets (e.g., processed combined CSVs)
- Runtime path in API references `data/combined_data.csv`

**Recommendation:** move to one canonical versioned storage source (object store/lake) and keep notebooks/Drive for collaboration only, not production truth.

---

## Data Requirements

## 1) Input data

## A) For `/predict` (Duy)
**Required request body:**
- `year` (integer)
- `start_lat` (float, valid latitude)
- `start_lon` (float, valid longitude)

**Format:** JSON body over HTTP POST.  
**Source:** frontend map click + user-selected context.

## B) For `/route` (Uday)
**Required request body:**
- `start_idx` / `goal_idx`
- Route tuning params (`ev_range_km`, `k_neighbors`, `alpha_weather`, `beta_traffic`, etc.)

**Format:** JSON body over HTTP POST.  
**Source:** UI controls (currently Streamlit; can be migrated to React).

---

## Feature Engineering

## A) Duy feature engineering (runtime geospatial)
From `start_lat`, `start_lon`, API derives:
- nearest road segment length (`SHAPE_Length`)
- nearest EV distance (meters)
- EV count within 500m
- nearest weather station aggregates (`avg_temp`, `total_prcp`)

Method:
- Convert geometries to metric CRS (EPSG:32755)
- Use nearest-distance lookup for traffic/weather
- Radius count for EV station proximity

## B) Uday feature engineering (routing)
- Builds `Weather_Sensitivity_Score` fallback from `TMAX`, `TMIN`, `PRCP` if missing
- Normalizes weather score and traffic proxy
- Creates graph edges via K-nearest neighbors + EV range filter
- Computes weighted travel cost using weather/traffic coefficients

---

## Model Performance

## A) Uday documented training/evaluation
- README reports Sprint 3 Random Forest weather-sensitivity model achieved **R² ~ 1.0**.

## B) Duy documented training/evaluation
- `README.md` states notebook trains/evaluates a regression model for traffic volume, but no explicit metric values are listed in the repo-level readme.

## Required next step for production handover
- Add a formal model card for `ev_model.pkl` including:
  - train/validation split method
  - MAE/RMSE/R²
  - data period
  - known limitations

---

## Expected Output (UI Map)

## Duy React map output
After map click and predict:
- marker at selected location
- predicted value (kWh display)
- route/environment insight card containing:
  - distance to nearest EV station
  - EV stations within 500m
  - average temperature
  - total precipitation
  - road segment length

## Uday Streamlit output
After route request:
- map with route polyline and step markers
- route status/hint
- total route cost (minutes)

---

## Deployment Guide

## 1) Prerequisites
- Python 3.10+
- Node.js 18+
- Access to model artifact and CSV data
- Google Maps API key

## 2) Install
### Backend
```bash
pip install pandas numpy scikit-learn joblib flask flask-cors geopandas shapely fastapi uvicorn pydantic
```

### Frontend
```bash
npm install
```

## 3) Config
Set environment variables:
- `REACT_APP_API_URL`
- `REACT_APP_GOOGLE_MAPS_API_KEY`
- `MODEL_PATH` (recommended)
- `DATA_ROOT` (recommended)

## 4) Output
Expected deployed outputs:
- backend API endpoints reachable
- frontend map UI reachable
- successful `/predict` and `/route` responses
- UI renders prediction card and/or route polyline

---

## Execution Flow

## A) Prediction flow
1. User clicks map in React UI.
2. Frontend captures `lat/lon`.
3. Frontend calls `POST /predict`.
4. Backend validates request.
5. Backend derives geospatial/weather features.
6. Model predicts result.
7. Backend returns JSON.
8. Frontend renders prediction and insights card.

## B) Routing flow
1. User selects start/goal and route parameters.
2. Frontend/Streamlit calls `POST /route`.
3. Backend loads/prepares nodes and edges.
4. Cost function applies weather + traffic weighting.
5. Dijkstra computes best path with stop penalties.
6. API returns path + total cost + status/hint.
7. UI renders route polyline and markers.

---

## Troubleshooting

1. **Server not reachable**
   - Confirm backend process is running.
   - Check `API_BASE_URL` in frontend.

2. **CORS error**
   - Ensure `flask-cors` is installed or proper CORS headers configured.
   - Restrict to allowed production domains when deployed.

3. **Model load failure**
   - Verify `ev_model.pkl` path exists and is readable.
   - Confirm scikit-learn compatibility version.

4. **Map not loading**
   - Validate Google Maps API key and billing status.

5. **No route found (`unreachable`)**
   - Increase `ev_range_km` and/or `k_neighbors`.
   - Choose closer start/goal indices.

---

## API Timeout - How to Solve

1. Increase backend request timeout limits at gateway/load balancer.
2. Add worker concurrency (Gunicorn/Uvicorn workers).
3. Cache heavy geospatial lookups (nearest neighbor indices).
4. Preload static dataframes/geodata at startup.
5. Add async job mode for long computations.
6. Add circuit-breaker/retry policy in frontend client.

---

## Future Enhancements

1. Replace Uday fallback adapter with live weather/traffic providers.
2. Merge routing + prediction into one versioned backend API.
3. Add route-by-coordinate API (instead of index-only routes).
4. Add authentication, rate limiting, and structured observability.
5. Add model card + continuous evaluation pipeline.
6. Add CI contract tests and e2e UI tests.

---

## Appendix A - Complete Feature List

## A) Duy `FEATURES` (model input)
| Feature | Description |
|---|---|
| `Year` | Request year |
| `SHAPE_Length` | Nearest road segment length |
| `dist_to_nearest_ev_m` | Distance to nearest EV station (m) |
| `ev_within_500m` | Count of EV stations within 500m |
| `avg_temp` | Average temperature from nearest station aggregate |
| `total_prcp` | Total precipitation from nearest station aggregate |

## B) Uday routing signals
| Signal | Description |
|---|---|
| `Weather_Score_Norm` | Normalized weather sensitivity score |
| `Traffic_Proxy_Norm` | Normalized traffic proxy |
| `EV_RANGE_KM` | Max edge feasibility distance |
| `K_NEIGHBORS` | Graph connectivity parameter |
| `ALPHA_WEATHER` | Weather impact coefficient |
| `BETA_TRAFFIC` | Traffic impact coefficient |
| `CHARGE_TIME_MIN` | Stop/charge penalty per intermediate stop |

---

## Appendix B - Sample API Responses

## 1) `GET /health` (routing service)
```json
{
  "status": "ok",
  "nodes": 300
}
```

## 2) `GET /stations`
```json
[
  {
    "node_id": "A001",
    "lat": -37.8123,
    "lon": 144.9671,
    "Weather_Score_Norm": 0.32,
    "Traffic_Proxy_Norm": 0.55
  }
]
```

## 3) `POST /route` success
```json
{
  "status": "ok",
  "hint": null,
  "params": {
    "EV_RANGE_KM": 70,
    "K_NEIGHBORS": 12,
    "ASSUMED_SPEED_KMH": 60,
    "ALPHA_WEATHER": 0.15,
    "BETA_TRAFFIC": 0.1,
    "CHARGE_TIME_MIN": 15,
    "MODE": "FALLBACK"
  },
  "nodes_count": 300,
  "edges_count": 1800,
  "path": [
    {"step": 0, "node_index": 0, "node_id": "A001", "lat": -37.81, "lon": 144.96},
    {"step": 1, "node_index": 9, "node_id": "A010", "lat": -37.82, "lon": 144.98}
  ],
  "total_cost_min": 41.2
}
```

## 4) `POST /route` unreachable
```json
{
  "status": "unreachable",
  "hint": "Increase ev_range_km or k_neighbors; or choose closer indices.",
  "path": [],
  "total_cost_min": null
}
```

## 5) `POST /predict` success
```json
{
  "year": 2026,
  "start_lat": -37.8136,
  "start_lon": 144.9631,
  "dist_to_nearest_ev_m": 120.3,
  "ev_within_500m": 3,
  "avg_temp": 15.4,
  "total_prcp": 812.0,
  "used_SHAPE_Length": 18.5,
  "prediction": 7.23
}
```

## 6) `POST /predict` validation error
```json
{
  "error": "Missing required fields: year, start_lat, start_lon"
}
```

