**This repository contains two complementary but separate implementations under:**

- Use_Cases/Weather-Aware Routing/Uday kiran reddy Neerudu/
- Use_Cases/Weather-Aware Routing/Duy Pham/

**They are not fully integrated yet.**

Uday’s track is route optimization-oriented; Duy’s track is web app + prediction-oriented.

**2) What Each Folder Is Trying to Solve**

2.1 Uday folder — routing intelligence prototype

Goal: Build weather-aware EV route planning logic and expose it through API + Streamlit.

Key components

- Routing core (engine sprint 4.py) with graph construction + weighted travel cost + Dijkstra + stop penalty.
- FastAPI wrapper (main - sprint 4.py) exposing /health, /stations, /route.
- Streamlit UI (ui_app sprint 4.py) consuming /stations and /route.

**Team note**

This is the closest artifact to “actual weather-aware routing.”

**2.2 Duy folder — prediction-focused web stack**

Goal: Provide map-based UX + backend prediction service for selected locations.

Key components

- Flask API (api.py) accepts year/start_lat/start_lon, derives geospatial features, predicts with ev_model.pkl, returns JSON.
- React + Google Maps frontend (MapComponent.tsx) click map → call API → display prediction and context metrics.
- Typed API client with input validation and error handling (apiClient.ts).

**Team note**

This is the strongest foundation for production web UX and API-client patterns.

**3) Current State of Integration (Important)**

**There is no single unified backend that combines:**

- Uday’s route planning endpoint(s), and
- Duy’s React app flow.

Duy frontend currently calls /predict (Flask), while Uday routing lives behind /route (FastAPI/engine path).

**4) API Contracts (as-is)**

**4.1 Uday API (routing)**

Endpoints

- GET /health
- GET /stations
- POST /route

Route request controls

Includes start_idx, goal_idx, ev_range_km, k_neighbors, alpha_weather, beta_traffic, charge_penalty_min, etc.

Route response shape

Returns status, hint, params, nodes_count, edges_count, path, total_cost_min.

**4.2 Duy API (prediction)**

Endpoints

- GET /
- POST /predict

Required request fields

year, start_lat, start_lon with validation and 400 errors for missing/invalid input.

Response fields

Includes derived features + prediction value.

**5) Recommended Combined Architecture (Target)**

**5.1 Product direction**

- For global deployment of a weather-aware routing product:
- Frontend: use Duy React map shell and API client style.
- Backend routing logic: use Uday route engine core.
- Optional: retain Duy prediction endpoint as additional analytics widget.

**5.2 Service layout**

- routing-service (Python): exposes /route, /stations, maybe /predict too.
- web-service (React static build + CDN): calls backend via env-configured API base URL.

**6) Detailed Notes for Dev Team**

**6.1 Data and feature computation notes**

- Uday engine auto-derives weather score and traffic proxy if missing; this is robust for imperfect datasets.
- Duy backend computes nearest-road, nearest-EV, and nearest-weather station features using projected CRS distances.

**Action:** document final canonical data schema and keep preprocessing logic centralized.

**6.2 Routing quality notes**

- Edge feasibility depends on EV_RANGE_KM + K_NEIGHBORS; unreachable routes return hints.
- Route cost function is tunable (alpha_weather, beta_traffic, charge penalty).

**Action:** calibrate these parameters with domain feedback and A/B tests.

**6.3 Frontend integration notes**

- Duy’s client-side validation and typed contracts are good production patterns.
- UI currently hardcodes year: 2026; make this user-configurable or backend-defaulted.
- Google Maps API key is expected from env var.

**6.4 Backend production hardening notes**

- Current Duy server runs Flask debug mode (debug=True); production should use WSGI server and disable debugging.
- CORS fallback currently uses wildcard origin; lock to approved frontend domains in prod.
- Uday adapter is a placeholder returning defaults; replace with real weather/traffic data provider integration.

**7) Deployment Plan (Global)**

**Phase 1 — unify API surface**

1. Merge Uday route engine into one backend service.
2. Keep Duy /predict endpoint if needed.
3. Define a versioned API contract (/v1/route, /v1/predict, /v1/stations).

**Phase 2 — frontend integration**

1. Keep Duy React app as a shell.
2. Add a route-planning panel calling /route.
3. Render polyline + charging stops from route response path.

**Phase 3 — production readiness**

1. Containerize backend + frontend.
2. Configure env vars: API URL, map key, model/data paths.
3. Add CI checks:
- API smoke tests (pattern already shown by test_api.py).
1. Add monitoring, request logging, and error alerts.

**8) Risks and Gaps to Track**

- Not yet unified architecture between route and prediction stacks.
- Data/model dependency risk (ev_model.pkl, CSV availability).
- Live data gap in Uday adapters (currently fallback defaults).
- Security gap if wildcard CORS / debug server stays in production.
- Product scope clarity needed: prediction-only vs true path routing.

**9) Suggested “Definition of Done” for Integration**

A release is done when:

- React app can request and render a route from backend /route.
- /predict and /route both run in one deploy environment.
- CORS restricted to production domains.
- Health checks and smoke tests pass post-deploy.
- API docs (request/response examples) are published for frontend and QA.

**10) Quick command references from project docs**

- Backend start pattern and dependency intent documented in setup files.
- Frontend startup and key setup documented in app guide.

**Uday kiran reddy Neerudu**

**Sprint 1:**

1. Load dataset
2. Preprocess the data , missing values filled with mean **due to??**
3. **Weather Sensitivity Scoring Logic**
    1. Making sure it is numeric
    2. Range is Tmax-Tmin, ensure that is in range, if not fill with 0, due to?
    3. Temp range score
        1. X > 20 = 1.0
        2. 20<= x <= 10 = 0.5
        3. x<10 = 0.0
    4. Weather sensitivity
        1. 2.0(TMAX >35) + 1.5(TMin<5) +
