# Setup Guide

**Environmental Impact Analysis — calvin-linardy**  
**EVAT T1 2026**

---

## Prerequisites

- Python 3.11+
- pip

---

## Step 1 — Install dependencies

```bash
cd "Use_Cases/Environmental Impact Analysis/calvin-linardy"
pip install -r requirements.txt
```

---

## Step 2 — Train the model

The XGBoost real-world adjustment model must be trained before the API can serve
predictions. This generates 8,000 synthetic training samples based on physics
literature and saves the model to `models/rw_adjustment_xgb.pkl`.

```bash
python train.py
```

To see evaluation examples immediately after training:
```bash
python train.py --evaluate
```

Expected output:
```
Generating synthetic training data...
  8,000 samples generated.
Training XGBoost model...
  Hold-out   MAE : 0.02xxx
  Hold-out   RMSE: 0.03xxx
  Hold-out   R²  : 0.9xxx
  CV R²  (5-fold): 0.9xxx ± 0.00xxx
Model saved to models/rw_adjustment_xgb.pkl
```

---

## Step 3 — Start the API

```bash
uvicorn api.main:app --reload --port 8001
```

The API will be available at:
- **Swagger UI:** http://localhost:8001/api/environmental-impact/docs
- **ReDoc:**      http://localhost:8001/api/environmental-impact/redoc
- **Health:**     http://localhost:8001/api/environmental-impact/health

---

## Step 4 — Test a prediction

### Using curl

**Database lookup (recommended):**
```bash
curl -X POST http://localhost:8001/api/environmental-impact/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ev": {
      "make": "Tesla",
      "model": "Model 3",
      "year": 2024,
      "variant": "Standard Range RWD"
    },
    "ice": {
      "make": "Toyota",
      "model": "RAV4",
      "year": 2024,
      "variant": "GX Petrol AWD"
    },
    "state": "VIC",
    "usage": {
      "annual_km": 15000,
      "driving_style": "mixed",
      "vehicle_age_years": 0
    },
    "include_lifecycle": true
  }'
```

**Manual entry (for vehicles not in database):**
```bash
curl -X POST http://localhost:8001/api/environmental-impact/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ev": {
      "consumption_kwh_per_100km": 17.5,
      "battery_kwh": 75.0
    },
    "ice": {
      "consumption_l_per_100km": 8.5,
      "fuel_type": "Diesel"
    },
    "state": "NSW",
    "usage": {
      "driving_style": "highway"
    },
    "include_lifecycle": true
  }'
```

### Browse available vehicles

```bash
# List EV makes
curl http://localhost:8001/api/environmental-impact/vehicles/ev

# List EV models for a make
curl http://localhost:8001/api/environmental-impact/vehicles/ev/Tesla/models

# List ICE makes
curl http://localhost:8001/api/environmental-impact/vehicles/ice

# List ICE models for a make
curl http://localhost:8001/api/environmental-impact/vehicles/ice/Toyota/models
```

---

## Step 5 — Explore notebooks

Open Jupyter and run the notebooks in order:

```bash
jupyter notebook notebooks/
```

| Notebook | What it covers |
|----------|---------------|
| `01_Data_Exploration.ipynb` | Vehicle data, grid factors, impact of state on savings |
| `02_Model_Training_Evaluation.ipynb` | XGBoost training, model comparison, feature importance |

---

## API Response Structure

```json
{
  "ev": {
    "wltp_consumption_kwh_per_100km": 14.9,
    "real_world_consumption_kwh_per_100km": 16.2,
    "grid_intensity_g_per_kwh": 990.0,
    "operational_g_per_km": 160.4,
    "manufacturing_g_per_km": 18.7,
    "total_g_per_km": 179.1
  },
  "ice": {
    "consumption_l_per_100km": 7.2,
    "fuel_type": "Petrol",
    "emission_factor_g_per_l": 2289.0,
    "operational_g_per_km": 164.8
  },
  "savings": {
    "co2_savings_g_per_km": -14.3,
    "co2_savings_kg_per_year": -187.3,
    "co2_savings_tonnes_lifetime": -2.81,
    "percentage_reduction": -8.7
  },
  "financial": {
    "ev_cost_aud_per_km": 0.0486,
    "ice_cost_aud_per_km": 0.144,
    "cost_saving_aud_per_km": 0.0954,
    "cost_saving_aud_per_year": 1249.7,
    "cost_saving_aud_lifetime": 18745.5
  },
  "equivalents": {
    "trees_planted_equivalent_per_year": 0,
    "petrol_litres_saved_per_year": 0.0
  },
  "state": "VIC",
  "driving_style": "mixed",
  "real_world_adjustment_factor": 1.087,
  "include_lifecycle": true,
  "ev_make": "Tesla",
  "ev_model": "Model 3",
  "ice_make": "Toyota",
  "ice_model": "RAV4"
}
```

> **Note:** A negative `co2_savings_g_per_km` in VIC means the EV (with lifecycle)
> emits *more* CO2 than the ICE in that scenario — this is intentionally honest.
> In VIC (990 g/kWh grid), an EV charged from the coal grid can emit more than
> an efficient petrol car when manufacturing emissions are included.
> The cost savings are still large because electricity is cheaper than petrol.

---

## File Structure

```
calvin-linardy/
├── README.md               — Project overview and API reference
├── SETUP_GUIDE.md          — This file
├── requirements.txt        — Python dependencies
├── train.py                — CLI to train and save the XGBoost model
├── data/
│   ├── README.md           — Data source documentation & update guide
│   ├── ev_vehicles.csv     — 58 EV models (Green Vehicle Guide 2024)
│   ├── ice_vehicles.csv    — 65 ICE vehicles (Green Vehicle Guide 2024)
│   ├── grid_emission_factors.csv  — State grid intensity (DCCEEW 2023)
│   └── fuel_emission_factors.csv  — Fuel CO2 factors (DCCEEW 2023)
├── src/
│   ├── config.py           — All constants (emission factors, defaults)
│   ├── data_loader.py      — CSV loading and vehicle lookup
│   ├── calculator.py       — Physics-based CO2 calculation engine
│   └── model.py            — XGBoost training and inference
├── api/
│   ├── schemas.py          — Pydantic request/response models
│   └── main.py             — FastAPI application
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   └── 02_Model_Training_Evaluation.ipynb
└── models/
    └── rw_adjustment_xgb.pkl   — Generated after running train.py
```

---

## Dataset Download Instructions

The four CSV files are already in `data/` and ready to use. If you want to
expand them with more vehicles:

### Green Vehicle Guide (EV and ICE data)
1. Go to: https://www.greenvehicleguide.gov.au/
2. Use the search filters:
   - **Fuel type:** Electric (for EVs) or Petrol/Diesel (for ICE)
   - **Year:** 2020–2024 recommended
3. For each result, record:
   - Make, Model, Year, Variant, BodyStyle, Segment
   - For EV: Energy Consumption (Wh/km) → divide by 10 for kWh/100km
   - For ICE: Combined Fuel Consumption (L/100km), Fuel Type
4. Add rows to the relevant CSV following the existing column order

### DCCEEW Emission Factors (grid and fuel)
1. Go to: https://www.dcceew.gov.au/climate-change/publications/national-greenhouse-accounts-factors
2. Download the latest "National Greenhouse Accounts Factors" document
3. Find:
   - **Table 1** — Fuel combustion emission factors (update `fuel_emission_factors.csv`)
   - **Table 3** — Electricity emission factors by state (update `grid_emission_factors.csv`)
4. Update the `Year` column and factor values accordingly

> The DCCEEW document is published annually, usually in August.
> Updating these factors annually will keep the model current as Australia's
> grid continues to decarbonise (especially VIC, NSW, QLD shifting toward renewables).

---

## Integration into EVAT-App-BE

Register this endpoint in the backend:

```
POST /api/environmental-impact/predict
```

The backend should:
1. Start this FastAPI app as a microservice (or integrate `src/calculator.py` directly)
2. Forward the request body as-is — no pre-computation required
3. Return the full response to the frontend

See `README.md` Section 6 for the complete integration plan.
