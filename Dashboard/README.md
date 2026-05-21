# EV Charging Station Dashboard with Congestion Prediction

This dashboard now runs with Dashboard-local congestion and price APIs plus centralized generic folders so additional use cases can be integrated under the same structure.

## Generic Folder Layout

```text
Dashboard/
  apis/
    congestion_prediction/
      app.py
    price_prediction/
      app.py
  data/
    EVAT.chargers.csv
    congestion_prediction/
      train_exogenous_3h.csv
  models/
    congestion_prediction/
      random_forest_model.pkl
  app_config.py
  dashboard.py
  run_stack.py
  requirements.txt
```

## Prerequisites

Before you begin, ensure Python is installed.

### Step 1: Create and Activate a Virtual Environment

1. Navigate to the Dashboard directory:

```bash
cd Dashboard
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration (Optional)

You can run with defaults, or configure ports and paths through a local .env file.

```bash
cp .env.example .env
```

Supported variables:

- EVAT_API_HOST
- EVAT_API_PORT
- EVAT_DASHBOARD_PORT
- EVAT_API_BASE_URL
- EVAT_STATIONS_DATA_PATH
- EVAT_CONGESTION_MODEL_PATH
- EVAT_CONGESTION_TRAIN_DATA_PATH
- PRICE_API_HOST
- PRICE_API_PORT
- PRICE_API_BASE_URL
- PRICE_MODEL_PATH
- PRICE_DATA_PATH
- PRICE_ALT_DATA_PATH
- PRICE_FEATURE_DICT_PATH

## How to Run

### Preferred: One Command (API + Dashboard)

From Dashboard directory:

```bash
python run_stack.py
```

This starts:

1. Congestion API (FastAPI)
2. Price Prediction API (FastAPI)
3. Dashboard UI (Streamlit)

### Alternative: Run Components Separately

Terminal 1:

```bash
uvicorn apis.congestion_prediction.app:app --reload --port 8000

uvicorn apis.price_prediction.app:app --reload --port 8001
```

Terminal 2:

```bash
streamlit run dashboard.py
```

## Using the Dashboard

1. Enter a postcode and click Find Stations.
2. Select prediction date/time.
3. Click Predict Congestion for All Stations.

Marker colors:

- Green: Low congestion
- Yellow: Medium congestion
- Red: High congestion
- Blue: Initial station state
- Grey: Unknown

## Migration Note

Congestion API code and required runtime datasets/models are now available directly under Dashboard. Legacy files under Use_Cases/Congestion Prediction remain unchanged for transition safety.
