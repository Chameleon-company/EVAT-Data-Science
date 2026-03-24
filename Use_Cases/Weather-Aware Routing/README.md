# Weather-Aware Routing & EV Prediction - Handover Documentation

**Document Version:** 1.0  
**Last Updated:** March 24, 2026  
**Scope:** `Use_Cases/Weather-Aware Routing/`  
**Primary Goal:** 
---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Current State](#current-state)
3. [System Architecture](#system-architecture)
4. [Repository Map](#repository-map)
5. [Key APIs](#key-apis)
6. [Data Sources & Source-of-Truth Policy](#data-sources--source-of-truth-policy)
7. [Dev Team Responsibilities (RACI)](#dev-team-responsibilities-raci)
8. [Implementation Plan](#implementation-plan)
9. [Deployment Guide (Global)](#deployment-guide-global)
10. [Testing, Monitoring & Operations](#testing-monitoring--operations)
11. [Risks, Gaps, and Decisions Needed](#risks-gaps-and-decisions-needed)
12. [Runbook (Quick Start)](#runbook-quick-start)
13. [Handover Checklist](#handover-checklist)

---

## Executive Summary

This use case currently has **two complementary implementations**:

- **Uday implementation**: route optimization prototype (graph + Dijkstra + weather/traffic penalties) exposed through FastAPI and demonstrated with Streamlit.
- **Duy implementation**: Flask prediction API + React Google Maps frontend for coordinate-based inference and visualization.

These tracks are **not yet unified** into a single production-grade deployment. The recommended path is:

1. Keep Duy's React app pattern for frontend UX.
2. Integrate Uday's route planning logic as the canonical routing service.
3. Keep prediction endpoint as optional analytics feature.
4. Establish a formal data source-of-truth and release process.

---

## Current State

### What exists
- Route planning API contract and routing engine prototype (Uday).
- Prediction API contract and web-map UX (Duy).
- Setup and integration guides for local execution.

### What is missing
- A single backend service exposing unified route + prediction APIs.
- Production-grade environment configuration and service hardening.
- Canonical, versioned, shared data source policy (beyond local CSV assumptions).
- Ownership and SLA definitions.

---

## System Architecture

```text
┌───────────────────────────────────────────────────────────────────┐
│                           Frontend (React)                       │
│  - Google Maps UI                                                 │
│  - API client + request validation                                │
│  - Route and prediction result rendering                          │
└───────────────────────────────┬───────────────────────────────────┘
                                │ HTTPS/JSON
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                         Backend API Layer                         │
│  /v1/health   /v1/stations   /v1/route   /v1/predict             │
│  - Request validation                                               │
│  - Authentication (future)                                         │
│  - CORS + rate limiting (prod)                                     │
└───────────────────────────────┬───────────────────────────────────┘
                                │
             ┌──────────────────┴──────────────────┐
             ▼                                     ▼
┌────────────────────────────┐         ┌────────────────────────────┐
│ Routing Engine             │         │ Prediction Engine          │
│ - Node/edge construction   │         │ - Geospatial features      │
│ - Weather/traffic weighting│         │ - ML model inference       │
│ - Dijkstra + stop penalty  │         │ - Feature output           │
└───────────────┬────────────┘         └───────────────┬────────────┘
                │                                      │
                ▼                                      ▼
         Canonical data store                    Canonical model store
```

---

## Repository Map

### Uday track (routing-focused)
- `main - sprint 4.py` - FastAPI endpoints and request schema.
- `engine sprint 4.py` - route planning core.
- `adapters sprint 4.py` - weather/traffic adapter scaffold.
- `utils -sprint 4.py` - geospatial/math helpers.
- `ui_app sprint 4.py` - Streamlit prototype UI.

### Duy track (web app + prediction-focused)
- `api.py` - Flask API + model inference.
- `apiClient.ts` - typed frontend API client.
- `MapComponent.tsx` - map click UX + prediction display.
- `useEVPrediction.ts` - reusable hook abstraction.
- `App.tsx` + `src/index.tsx` - application shell/entry point.
- `README_APP.md`, `INTEGRATION_GUIDE.md`, `SETUP_GUIDE.md` - setup docs.

---

## Key APIs

## 1) Routing API (target canonical)

### Endpoint
`POST /v1/route`

### Purpose
Compute weather-aware, traffic-aware EV route between selected nodes.

### Request (based on Uday prototype)
```json
{
  "start_idx": 0,
  "goal_idx": 10,
  "ev_range_km": 35.0,
  "k_neighbors": 6,
  "assumed_speed_kmh": 60.0,
  "alpha_weather": 0.15,
  "beta_traffic": 0.10,
  "charge_penalty_min": 15.0,
  "mode": "FALLBACK"
}
```

### Response (based on Uday engine)
```json
{
  "status": "ok",
  "hint": null,
  "params": { "...": "..." },
  "nodes_count": 123,
  "edges_count": 456,
  "path": [
    { "step": 0, "node_index": 0, "node_id": "A", "lat": -37.81, "lon": 144.96 }
  ],
  "total_cost_min": 42.5
}
```

### Related endpoints
- `GET /v1/stations` - station/node list.
- `GET /v1/health` - service health.

---

## 2) Prediction API (optional but recommended)

### Endpoint
`POST /v1/predict`

### Purpose
Given a coordinate + year, return predicted value and contributing geospatial/weather context.

### Request (based on Duy API)
```json
{
  "year": 2026,
  "start_lat": -37.8136,
  "start_lon": 144.9631
}
```

### Response (based on Duy API)
```json
{
  "year": 2026,
  "start_lat": -37.8136,
  "start_lon": 144.9631,
  "dist_to_nearest_ev_m": 120.0,
  "ev_within_500m": 3,
  "avg_temp": 9.4,
  "total_prcp": 820.0,
  "used_SHAPE_Length": 18.5,
  "prediction": 7.23
}
```

---

## 3) Error contract (recommended standard)

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Missing required fields: year, start_lat, start_lon",
    "details": {}
  }
}
```

HTTP status:
- 400 validation error
- 404 resource not found
- 429 rate-limited
- 500 internal error

---

## Data Sources & Source-of-Truth Policy

## Current data usage
- Traffic CSV
- EV station CSV
- Weather CSV (including weather-by-station aggregates)
- Combined processed CSV for routing
- Trained model artifact (`ev_model.pkl`)


---

## Deployment Guide (Global)

## Environment split
- **Dev**: local containers, sample datasets.
- **Staging**: production-like infra with masked/prod-like data.
- **Prod**: auto-deploy from tagged release.

## Deployment artifacts
- Backend container image (API + routing/prediction modules).
- Frontend static build artifact.
- Model artifact bundle with explicit version.
- Dataset manifest (immutable IDs/paths).

## Required environment variables
- `API_PORT`
- `CORS_ALLOWED_ORIGINS`
- `DATASET_ROOT_URI`
- `MODEL_URI`
- `REACT_APP_API_URL`
- `REACT_APP_GOOGLE_MAPS_API_KEY`

## Security
- HTTPS only.
- Secret management via platform vault.
- Least-privilege access to data/model buckets.

---

## Testing, Monitoring & Operations

## Minimum tests
1. API health tests.
2. Contract tests for `/route`, `/stations`, `/predict`.
3. Regression tests on fixed coordinate/node scenarios.
4. Frontend e2e tests (map click -> API call -> render).

## Monitoring
- Latency p50/p95/p99 by endpoint.
- Error rate by HTTP code.
- Route success/unreachable ratios.
- Prediction drift indicators (if labels available later).

## Operational alerts
- API 5xx spike.
- p95 latency breach.
- Model/data artifact load failure.
- CORS/auth failures at scale.

---

## Risks, Gaps, and Decisions Needed

## Current risks
- Two parallel stacks without unified contract.
- Placeholder adapter logic for weather/traffic in routing path.
- Local-file assumptions for data/model loading.
- Potential mismatch between prediction and routing release cadence.

## Decisions needed now
1. Single-service vs multi-service backend architecture.
2. Keep prediction endpoint in MVP or defer.
3. Canonical data platform and versioning method.
4. Release cadence ownership (who signs off backend/model/data changes).

---

## Runbook (Quick Start)

## Local backend smoke
1. Install dependencies.
2. Start backend service.
3. `GET /health` and `GET /` checks.
4. `POST /predict` test request.
5. `POST /route` test request.

## Local frontend smoke
1. Set `REACT_APP_GOOGLE_MAPS_API_KEY`.
2. Set `REACT_APP_API_URL`.
3. Start app and click map.
4. Verify request/response in browser dev tools.

---

## Handover Checklist

- [ ] API v1 contract approved and documented.
- [ ] Backend service unified and deployable.
- [ ] Frontend connected to final API endpoints.
- [ ] Data/model source-of-truth policy implemented.
- [ ] CI/CD and release process live.
- [ ] Monitoring dashboards and alerts active.
- [ ] Operational runbook tested by on-call engineer.
- [ ] Knowledge transfer session completed and recorded.

---

## Appendix A - Suggested Ownership by Existing Work

- **Routing logic owner candidate:** contributor familiar with `engine sprint 4.py`.
- **Frontend integration owner candidate:** contributor familiar with `MapComponent.tsx` and `apiClient.ts`.
- **Prediction model owner candidate:** contributor maintaining `api.py` + model artifact.
- **Platform owner:** DevOps/Platform team to own deployment and observability.

