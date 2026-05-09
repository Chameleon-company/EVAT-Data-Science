# Price Prediction API + Dashboard

This folder contains a FastAPI service and a Streamlit dashboard for the price prediction model.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn price_prediction_api:app --reload --port 8001
```

## Run the Dashboard

```bash
streamlit run price_prediction_dashboard.py --server.port 8502
```

## Configuration

You can override paths and ports with environment variables:

- `PRICE_API_HOST` (default: 127.0.0.1)
- `PRICE_API_PORT` (default: 8001)
- `PRICE_DASHBOARD_PORT` (default: 8502)
- `PRICE_API_BASE_URL` (default: http://127.0.0.1:8001)
- `PRICE_MODEL_PATH` (default: artifacts/price_best_model_latest.joblib)
- `PRICE_DATA_PATH` (default: artifacts/car_price_enriched_latest.csv)
- `PRICE_ALT_DATA_PATH` (default: car_price_prediction_enriched_features.csv)
- `PRICE_FEATURE_DICT_PATH` (default: artifacts/feature_dictionary.csv)
