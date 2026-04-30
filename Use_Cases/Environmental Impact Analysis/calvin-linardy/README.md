# Environmental Impact Analysis

**calvin-linardy | EVAT T1 2026**

Estimates CO₂ savings (and financial savings) when replacing an ICE vehicle with an EV.
Designed for integration into the EVAT platform as a user-facing feature.

---

## What this does

A user picks an EV and an ICE vehicle. The system returns:
- How many grams of CO₂ per km they would save (or emit extra, if the grid is dirty)
- Annual and lifetime CO₂ impact
- Annual running cost saving in AUD
- How many trees that equates to per year

---

## Why this is different from the previous model

The previous implementation (Ruvinya-Ekanayake) had several critical flaws:

| Problem | Impact |
|---------|--------|
| Hardcoded grid factor: 0.18 kg/kWh (SA's figure) | Overstated savings by 3–7× in NSW, VIC, QLD |
| GradientBoost applied to deterministic arithmetic | ML was learning algebra, not real-world patterns |
| 200-row datasets, Cartesian join, 4K sample | Unrealistic vehicle pairings |
| Caller must pre-compute ICE CO₂ baseline | No documented standard — high error risk |
| No lifecycle emissions | Incomplete picture |

This implementation fixes all of these.

---

## Design

```
User Input (EV + ICE + state + usage)
          │
          ▼
 Vehicle Lookup (data/)
 → EV: kWh/100km, battery kWh
 → ICE: L/100km, fuel type
          │
          ▼
 XGBoost Adjustment Model
 → Predicts real-world deviation from WLTP
   based on: temperature, driving style,
             vehicle age, battery size
          │
          ▼
 Physics Calculator (src/calculator.py)
 ┌────────────────────────────────────┐
 │ EV: kWh/100km × RW_factor         │
 │   × grid_g_CO2/kWh (by state)     │
 │   + battery mfg amortised         │
 │                                    │
 │ ICE: L/100km                       │
 │   × fuel_g_CO2/L (DCCEEW 2023)    │
 │                                    │
 │ Savings = ICE − EV                 │
 └────────────────────────────────────┘
          │
          ▼
 Rich JSON Response
 → CO₂ savings (g/km, kg/yr, t/lifetime)
 → Financial savings (AUD/yr, AUD/lifetime)
 → Equivalents (trees, petrol litres)
```

---

## Emission factors used

All sourced from DCCEEW National Greenhouse Accounts Factors 2023.

### Grid (electricity)

| State | g CO₂/kWh |
|-------|-----------|
| NSW   | 790       |
| ACT   | 0         |
| VIC   | 990       |
| QLD   | 810       |
| SA    | 290       |
| WA    | 650       |
| TAS   | 130       |
| NT    | 590       |

### Fuel

| Fuel    | kg CO₂/L |
|---------|----------|
| Petrol  | 2.289    |
| Diesel  | 2.703    |
| E10     | 2.195    |
| LPG     | 1.542    |

---

## Quick start

```bash
pip install -r requirements.txt
python train.py --evaluate       # train model, print example predictions
uvicorn api.main:app --reload --port 8001
```

Full setup instructions: **SETUP_GUIDE.md**

---

## API

```
POST /api/environmental-impact/predict
GET  /api/environmental-impact/vehicles/ev
GET  /api/environmental-impact/vehicles/ev/{make}/models
GET  /api/environmental-impact/vehicles/ice
GET  /api/environmental-impact/vehicles/ice/{make}/models
GET  /api/environmental-impact/health
```

Interactive docs: http://localhost:8001/api/environmental-impact/docs

### Example request

```json
POST /api/environmental-impact/predict

{
  "ev": {
    "make": "Tesla",
    "model": "Model 3",
    "year": 2024
  },
  "ice": {
    "make": "Toyota",
    "model": "RAV4",
    "year": 2024
  },
  "state": "NSW",
  "usage": {
    "annual_km": 13100,
    "driving_style": "mixed",
    "vehicle_age_years": 0
  },
  "include_lifecycle": true
}
```

**Or pass raw consumption directly (for any vehicle not in the database):**

```json
{
  "ev": { "consumption_kwh_per_100km": 17.5, "battery_kwh": 75.0 },
  "ice": { "consumption_l_per_100km": 8.5, "fuel_type": "Diesel" },
  "state": "QLD"
}
```

### Example response (abbreviated)

```json
{
  "savings": {
    "co2_savings_g_per_km": 21.4,
    "co2_savings_kg_per_year": 280.3,
    "co2_savings_tonnes_lifetime": 4.2,
    "percentage_reduction": 12.3
  },
  "financial": {
    "cost_saving_aud_per_year": 1240.0,
    "cost_saving_aud_lifetime": 18600.0
  },
  "equivalents": {
    "trees_planted_equivalent_per_year": 13,
    "petrol_litres_saved_per_year": 122.5
  },
  "state": "NSW",
  "real_world_adjustment_factor": 1.082,
  "include_lifecycle": true
}
```

---

## File structure

```
calvin-linardy/
├── README.md
├── SETUP_GUIDE.md
├── requirements.txt
├── train.py                    ← run this first
├── data/
│   ├── README.md               ← how to update datasets
│   ├── ev_vehicles.csv         ← 58 EVs (Green Vehicle Guide 2024)
│   ├── ice_vehicles.csv        ← 65 ICE vehicles (Green Vehicle Guide 2024)
│   ├── grid_emission_factors.csv  ← DCCEEW 2023
│   └── fuel_emission_factors.csv  ← DCCEEW 2023
├── src/
│   ├── config.py               ← emission factors, defaults, constants
│   ├── data_loader.py          ← CSV loading, vehicle lookup functions
│   ├── calculator.py           ← physics engine (deterministic)
│   └── model.py                ← XGBoost training & inference
├── api/
│   ├── schemas.py              ← Pydantic request/response models
│   └── main.py                 ← FastAPI app
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   └── 02_Model_Training_Evaluation.ipynb
└── models/
    └── rw_adjustment_xgb.pkl   ← generated by train.py
```

---

## Integration plan (EVAT-App-BE + EVAT-Website)

### Backend (EVAT-App-BE)

Register one new route:

```
POST /api/environmental-impact/predict
```

Options:
- **Option A (microservice):** Run this FastAPI app on port 8001 and proxy through EVAT-App-BE
- **Option B (direct import):** Import `src.calculator` and `src.model` directly into the BE codebase — no second process needed

The request/response schemas are in `api/schemas.py` (Pydantic v2).

### Frontend (EVAT-Website)

Minimum viable UI:
1. Dropdown to select Australian state
2. EV selector: make → model (populated from `GET /vehicles/ev` and `GET /vehicles/ev/{make}/models`)
3. ICE selector: make → model (same pattern)
4. Optional: annual km slider, driving style radio button
5. Display result cards: CO₂ saving, annual cost saving, tree equivalent

The vehicle dropdowns are powered by the vehicle listing endpoints so they always
reflect what is in the database — no hardcoded lists needed on the frontend.

---

## Data update cadence

| Data | Source | Update frequency |
|------|--------|-----------------|
| Grid factors | DCCEEW NGA Factors | Annually (August) |
| Fuel factors | DCCEEW NGA Factors | Annually (August) |
| EV vehicles | Green Vehicle Guide | As new models enter market |
| ICE vehicles | Green Vehicle Guide | As new models enter market |

See `data/README.md` for step-by-step update instructions.
