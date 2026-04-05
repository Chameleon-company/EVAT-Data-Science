# Environmental Impact Analysis — Integration Guide

*T1 2026*

---

# 1. What this use case is trying to do

The Environmental Impact Analysis use case estimates the CO₂ savings (in grams per kilometre) when replacing an Internal Combustion Engine (ICE) vehicle with an Electric Vehicle (EV). The model compares EV energy consumption against ICE fuel consumption and returns a predicted CO₂ saving value.

The end goal is a feature within the EVAT platform where a user selects an EV and an ICE vehicle, and the system displays the estimated environmental benefit of switching — supporting informed decision-making around EV adoption.

The model has been trained and tested locally. This document covers the current state of the use case, the integration gaps that remain, and the steps required to connect it to the EVAT backend and frontend.

---

# 2. Current State

The following table summarises what has been completed and what remains:

| Area | Status | Notes |
|------|--------|-------|
| ML model | Complete | `co2_savings_model.pkl` trained and serialised |
| Training pipeline | Complete | Clean, modular code in `Clean_Model_Code.ipynb` |
| API schema | Documented | Input/output schema defined in `api_schema.md` |
| FastAPI endpoint | Local only | Runs locally via uvicorn — not deployed to shared environment |
| Backend integration | Not started | No endpoint registered in EVAT-App-BE |
| Frontend integration | Not started | No UI component consuming this use case |

---

# 3. Data Sources

The model was trained on the following datasets, located in `Environment_Analysis/Data/`:

| Dataset | Description | Used For |
|---------|-------------|----------|
| `Pure electric consumption.csv` | EV energy usage (Wh/km) | EV CO₂ calculation |
| `Diesel consumption.csv` | Diesel ICE fuel consumption | ICE CO₂ baseline |
| `petrol91RON consumption.csv` | Petrol 91 ICE consumption | ICE CO₂ baseline |
| `petrol95RON consumption.csv` | Petrol 95 ICE consumption | ICE CO₂ baseline |
| `petrol98RON consumption.csv` | Petrol 98 ICE consumption | ICE CO₂ baseline |

These are static datasets used for model training. They are not loaded at prediction time — the trained model (`co2_savings_model.pkl`) is loaded directly for inference.

---

# 4. Core Calculations

Understanding these calculations is important for backend developers, particularly for pre-computing the `ICE_CO2_Baseline` field required by the API.

### EV Emissions
```
EV_gCO2_per_km = (EnergyConsumptionWh/km ÷ 1000) × 0.18 × 1000
```

### ICE Baseline
```
ICE_CO2_Baseline = FuelConsumptionCombined × EmissionFactor
```

### Emission Factors

| Fuel Type | Emission Factor | Unit |
|-----------|----------------|------|
| Petrol (all grades) | 23.2 | kg CO₂ per L |
| Diesel | 26.5 | kg CO₂ per L |
| Electricity | 0.18 | kg CO₂ per kWh |

These are fixed constants used during model training. They should be validated and potentially made configurable before production deployment.

### CO₂ Saving (Target Variable)
```
CO2_saving = ICE_CO2_Baseline − EV_gCO2_per_km
```

A positive value means the EV emits less CO₂ per kilometre than the ICE vehicle.

### Engineered Features

| Feature | Description |
|---------|-------------|
| `EV_gCO2_per_km` | EV CO₂ emissions derived from energy consumption |
| `ICE_CO2_Baseline` | ICE CO₂ emissions derived from fuel consumption |
| `YearDiff` | EV model year minus ICE model year |
| `CO2_saving` | Target variable — difference between ICE and EV emissions |

---

# 5. Model

Three models were evaluated using 5-fold cross-validation:

| Model | Mean R² | Mean MAE | Outcome |
|-------|---------|----------|---------|
| Linear Regression | Low | Moderate | Underfitting — rejected |
| Random Forest | Moderate | Good | Stable but not selected |
| Gradient Boosting | Best | Lowest | Selected |

**Final model:** `GradientBoostingRegressor` (scikit-learn)

**Production pipeline steps:**
1. Load and clean EV and ICE datasets
2. Compute emission features for each dataset
3. Combine EV and ICE using a Cartesian merge (sampled to 4,000 rows)
4. Engineer `YearDiff` and `CO2_saving` features
5. Encode categorical columns using `OneHotEncoder`
6. Train `GradientBoostingRegressor` and export as `co2_savings_model.pkl`

---

# 6. API Specification

### Endpoint
```
POST /predict
```

### Request

```json
{
  "Make_EV": "Tesla",
  "Make_ICE": "Toyota",
  "BodyStyle_EV": "SUV",
  "BodyStyle_ICE": "SUV",
  "FuelType_ICE": "Petrol95",
  "YearDiff": 5,
  "ICE_CO2_Baseline": 220.4
}
```

| Field | Type | Description |
|-------|------|-------------|
| `Make_EV` | string | Brand of the EV |
| `Make_ICE` | string | Brand of the ICE vehicle |
| `BodyStyle_EV` | string | EV body style (e.g. SUV, Sedan) |
| `BodyStyle_ICE` | string | ICE body style |
| `FuelType_ICE` | string | One of: `Petrol91`, `Petrol95`, `Petrol98`, `Diesel` |
| `YearDiff` | number | EV model year minus ICE model year |
| `ICE_CO2_Baseline` | number | Pre-calculated ICE CO₂ emissions (g/km) — must be computed by the caller |

### Response

```json
{
  "Predicted_CO2_Savings": 134.72
}
```

This means the EV is predicted to emit 134.72 g/km less CO₂ than the ICE vehicle.

---

# 7. Integration Gaps

The following issues must be resolved before this use case can be delivered end-to-end within the EVAT platform.

## 7.1 Critical

**No backend integration.** The EVAT-App-BE repository has no registered endpoint for this use case. The backend team needs to add a route that loads `co2_savings_model.pkl` and serves predictions via the EVAT API.

**No frontend integration.** The EVAT-Website has no UI component for this feature. Users currently have no way to access environmental impact analysis through the platform.

## 7.2 High Severity

**Local deployment only.** The FastAPI prediction service runs locally via uvicorn and is not accessible from any shared environment. This blocks both backend and frontend integration until a deployment solution is agreed upon.

**Hardcoded data paths.** `Clean_Model_Code.ipynb` references local absolute paths (e.g. `/content/Data/`). These must be updated to relative paths before the pipeline can be run consistently across team members.

**`ICE_CO2_Baseline` must be pre-computed by the caller.** This field is a required API input but the calculation logic currently lives only in the training notebook. This logic needs to be either documented clearly for the backend team or moved into the API itself.

## 7.3 Medium Severity

- No input validation or error handling defined for the API endpoint beyond the base schema
- String field values are case-sensitive — `"Petrol95"` is valid, `"petrol95"` will cause a prediction error
- Unseen vehicle makes (not present in training data) will reduce prediction accuracy without warning
- No batch prediction support — the API processes one EV–ICE pair per request

---

# 8. Integration Plan

### Step 1 — Resolve Local Dependencies
- Update data paths in `Clean_Model_Code.ipynb` to use relative paths
- Confirm `co2_savings_model.pkl` loads correctly from the expected directory

### Step 2 — Backend Integration (EVAT-App-BE)
- Register a new route: `POST /api/environmental-impact/predict`
- Accept the input schema defined in Section 6
- Load `co2_savings_model.pkl` and run inference
- Return the `Predicted_CO2_Savings` value as JSON
- Add input validation and error responses for malformed requests

### Step 3 — Frontend Integration (EVAT-Website)
- Build a UI component for EV and ICE vehicle selection
- Send a `POST` request to the backend endpoint on form submission
- Display the returned CO₂ savings value with appropriate context

### Step 4 — End-to-End Testing
- Validate the full flow: Frontend → Backend → Model → Backend → Frontend
- Test with edge cases (unknown vehicle makes, missing or mistyped fields)
- Confirm response values match locally tested results

---

# 9. System Architecture

```
User (Frontend)
    │
    ▼
EVAT-Website
Vehicle selection form
    │  POST /api/environmental-impact/predict
    ▼
EVAT-App-BE
Route handler + input validation
    │  Loads co2_savings_model.pkl
    ▼
Gradient Boosting Model
Returns Predicted_CO2_Savings
    │
    ▼
EVAT-App-BE
Formats and returns JSON response
    │
    ▼
EVAT-Website
Displays CO₂ savings to user
```

---

# 10. Related Files

| File | Description |
|------|-------------|
| `Ruvinya-Ekanayake/Clean_Model_Code.ipynb` | Production-ready model training pipeline |
| `Ruvinya-Ekanayake/co2_savings_model.pkl` | Trained and serialised Gradient Boosting model |
| `Ruvinya-Ekanayake/api_schema.md` | Input/output schema reference |
| `Ruvinya-Ekanayake/README.md` | T3 2025 model documentation |
| `Environment_Analysis/EVAT PROJECT ENVIRONMENTAL.ipynb` | Original model notebook (T2 2025) |
