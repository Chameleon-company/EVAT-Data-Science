# Home Charger Rental – API (OpenAPI Summary)

This folder contains an **OpenAPI specification** for the Home Charger Rental use case.

The goal is to provide a **decision-support API** (not a production-ready service yet) that helps the EVAT app and teammates quickly retrieve:
- **Suburb cluster segment** (business-friendly label)
- **Key summary metrics** used for interpretation
- **Indicative pricing signal** (e.g., recommended/optimal price per suburb)

> ✅ Target audience: mentors, API team, web team, and non-technical stakeholders  
> ✅ This API spec is meant for alignment and handover, not deployment

---

## What is inside

- `openapi.yaml`  
  OpenAPI 3.x specification describing endpoints, parameters, and JSON responses.

---

## What this API is meant to support in EVAT

In the EVAT app, the results can be used for:
- Showing suburb segments on UI (e.g., “EV-Ready”, “High Population – Infrastructure Gap”)
- Supporting planning decisions (where home charger rental might be higher demand)
- Displaying an indicative **pricing signal** (not a final price recommendation)

---

## Data sources behind the outputs (high-level)

The API responses are based on cleaned and merged suburb-level datasets:
- **Population dataset** (suburb population)
- **PCZ / suburb profile metrics** (dwellings, income, vehicles per dwelling)
- **Station / charger counts** (public chargers and station counts)
- **Congestion** (average congestion proxy, filled with median where not available)
- **Pricing output** (optimal price per suburb)

Cleaned outputs are stored under:
- `datasets/clean/`

---

## Endpoints overview (conceptual)

The `openapi.yaml` describes endpoints like:
- **Get suburb summary**
  - Input: suburb name (optional lat/lon)
  - Output: cluster id/label, key metrics, pricing signal
- **List suburbs by cluster**
  - Input: cluster id/label
  - Output: list of suburbs + summary stats
- **Get pricing signal**
  - Input: suburb
  - Output: optimal price and supporting info

> Note: Exact endpoint names/paths should be confirmed with the API team.

---

## Important notes / limitations

- This is a **prototype specification** for sprint handover.
- Some suburbs may have missing coordinates (lat/lon) due to limited coordinate coverage in the provided datasets.
- “Optimal_Price” is an **indicative signal** generated from a simplified pricing logic/model for the capstone stage.
- Results should be treated as **decision support**, not final policy or final pricing.

---

## Next step for the API team

If the backend team wants to implement this:
1. Confirm which outputs are required in the EVAT UI.
2. Decide minimal stable fields for the response:
   - `suburb`, `cluster_id`, `cluster_label`, `metrics`, `optimal_price`
3. Implement a lightweight service (FastAPI/Flask) that reads from:
   - `datasets/clean/master_suburb_table_with_congestion.csv`
   - `datasets/clean/optimal_pricing_by_suburb.csv`

---

## Owner / Sprint

Maintained by: **Daniel Nguyen**  
Sprint focus: **Sprint 3 – Data Handover & Documentation**
