# Price Prediction Model

End-to-end documentation for the EV price prediction model, including training workflow, artifacts, and serving API.

## Overview

The model predicts `Log_Price` and converts predictions back to `Price` with an inverse log transform. A FastAPI service exposes prediction endpoints, and a Streamlit dashboard provides a UI for exploration and batch runs.

## Data and Feature Pipeline

Source data is enriched with engineered features and optional external signals. The EDA notebook outlines the feature logic and quality checks.

Feature groups:
- Original dataset fields (for example `Brand`, `Model`, `Year`, `Mileage`).
- Engineered features: age, log mileage, mileage-per-year, condition score, and anomaly flags.
- Internet-enriched proxies: yearly weather averages, Brent oil price, AU CPI, and a composite energy pressure index.

The enriched dataset and a feature dictionary are exported to `artifacts/` for model training and serving.

## Training and Evaluation Workflow

Training is documented in the model development notebook and uses a consistent pre-processing pipeline:
- Numeric: median imputation + standard scaling.
- Categorical: most-frequent imputation + one-hot encoding.

Target definition and transforms:
- Training target is `Log_Price = log1p(Price)`.
- Evaluation converts predictions back to price: `Price = expm1(Log_Price)` with a floor at 0.

Data splits and randomness:
- Train/test split: 75% train, 25% test (fixed random seed 42).
- Train/validation split within training: 80% train, 20% validation.
- All model training uses `RANDOM_STATE = 42` where supported.

Model set and selection:
- Base models: LinearRegression, Ridge, Lasso, ElasticNet, KNN, RandomForest, GradientBoosting.
- Optional models if installed: XGBoost, LightGBM, CatBoost.
- Fallbacks (to keep 10 models): ExtraTrees, AdaBoost, HistGradientBoosting.

Hyperparameter tuning:
- Hyperopt (TPE) runs `MAX_EVALS = 30` trials per model.
- Validation metric for tuning: RMSE on price (after inverse log transform).

Evaluation:
- Each tuned model is re-fit on the full training set and scored on the held-out test set.
- Metrics on original price scale: MAE, RMSE, R2, MAPE.
- Per-row test predictions are saved for residual analysis.

Artifacts produced during training:
- `price_best_model_latest.joblib` (best model pipeline)
- `price_model_scorecard_latest.csv` (test metrics by model)
- `price_model_hyperopt_latest.csv` (tuning results)
- `price_model_predictions_latest.csv` (test predictions)
- `car_price_enriched_latest.csv` (enriched dataset)
- `feature_dictionary.csv` (feature schema and descriptions)

Scorecard results (`price_model_scorecard_latest.csv`):

| model | mae | rmse | r2 | mape_pct |
| --- | --- | --- | --- | --- |
| ElasticNet | 24753.925247115127 | 29202.618449952002 | -0.13375299664358398 | 78.82090837734413 |
| Lasso | 24756.193896541645 | 29216.213171599094 | -0.1348088364861284 | 78.8132829098775 |
| GradientBoosting | 25018.57396778808 | 29465.31148463866 | -0.15424215796513696 | 79.97231952409828 |
| ExtraTrees | 24954.985139403994 | 29468.574258365974 | -0.15449779683739773 | 79.1595237399068 |
| RandomForest | 24962.916925193134 | 29483.63001617598 | -0.1556777846669677 | 79.29924526943056 |
| Ridge | 24992.158494542604 | 29505.896848342963 | -0.15742404186058834 | 79.59214413965685 |
| XGBoost | 25094.85900475 | 29566.102459455557 | -0.16215221638400346 | 79.94119015758915 |
| LightGBM | 25195.881675379478 | 29795.051121677618 | -0.18022043355692263 | 80.89086033211919 |
| KNN | 25208.652227594368 | 29904.38340233491 | -0.18889791082493868 | 81.18353245071242 |
| LinearRegression | 25377.498020014915 | 29951.80061285721 | -0.19267119824068368 | 81.15527964614921 |

Hyperopt summary (`price_model_hyperopt_latest.csv`):

| model | val_rmse | best_params |
| --- | --- | --- |
| XGBoost | 27112.867097920247 | {"colsample_bytree": 0.7012165922971707, "learning_rate": 0.012555935566382383, "max_depth": 3.0, "min_child_weight": 0.24379375323724536, "n_estimators": 250.0, "reg_alpha": 0.9627549985316267, "reg_lambda": 0.17709202628393075, "subsample": 0.9561108770491097} |
| GradientBoosting | 27161.87081942159 | {"learning_rate": 0.04151960499529325, "max_depth": 2, "max_features": "sqrt", "min_samples_leaf": 4.0, "min_samples_split": 10.0, "n_estimators": 175.0, "subsample": 0.7978402967642961} |
| RandomForest | 27166.75675472413 | {"bootstrap": true, "max_depth": 4, "max_features": null, "min_samples_leaf": 5.0, "min_samples_split": 6.0, "n_estimators": 375.0} |
| Lasso | 27243.31642189476 | {"alpha": 0.008013435956877863} |
| ElasticNet | 27250.36923812002 | {"alpha": 0.025280510730854546, "l1_ratio": 0.3600259141883796} |
| ExtraTrees | 27251.573748402934 | {"bootstrap": true, "max_depth": 4, "max_features": null, "min_samples_leaf": 5.0, "min_samples_split": 7.0, "n_estimators": 350.0} |
| LightGBM | 27345.104852617736 | {"colsample_bytree": 0.6065760397460757, "learning_rate": 0.010405925487893576, "min_child_samples": 45.0, "n_estimators": 200.0, "num_leaves": 29.0, "subsample": 0.8295072584830219} |
| Ridge | 27415.289201639112 | {"alpha": 98.1808820968703} |
| LinearRegression | 27461.85819626928 | {"fit_intercept": true, "positive": false} |
| KNN | 27902.07191840866 | {"n_neighbors": 21.0, "p": 2, "weights": "distance"} |

## Quickstart

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn price_prediction_api:app --reload --port 8001
```

Run the dashboard:

```bash
streamlit run price_prediction_dashboard.py --server.port 8502
```

## API Usage

Base URL (default): `http://127.0.0.1:8001`

Endpoints:
- `GET /health`: service status and model load state
- `GET /schema`: feature schema (columns, numeric/categorical split, descriptions)
- `GET /model/info`: model type and pipeline steps
- `POST /predict`: single-row prediction
- `POST /predict/batch`: batch prediction

Schema behavior:
- The service derives the feature schema from the trained model when available (`feature_names_in_`).
- If the model does not expose feature names, it falls back to the header of `PRICE_DATA_PATH` or `PRICE_ALT_DATA_PATH`.
- `GET /schema` returns `feature_columns`, `numeric_columns`, `categorical_columns`, and `feature_descriptions` (from `feature_dictionary.csv` when present).

Prediction behavior:
- Inputs are normalized to the schema order; missing columns are filled with `NaN` and extra columns are ignored.
- Numeric columns are coerced with `pandas.to_numeric(errors="coerce")`.
- Output includes both `predicted_log_price` and `predicted_price` (inverse log transform, clipped at 0).

Error handling:
- `503` if the model or schema is not loaded.
- `500` during startup if the model artifact cannot be loaded.

Example curl requests:

```bash
# Health
curl -s http://127.0.0.1:8001/health

# Schema
curl -s http://127.0.0.1:8001/schema

# Model info
curl -s http://127.0.0.1:8001/model/info

# Single prediction
curl -s -X POST http://127.0.0.1:8001/predict \
	-H "Content-Type: application/json" \
	-d '{
		"row_id": "optional-id",
		"features": {
			"Brand": "Tesla",
			"Model": "Model 3",
			"Year": 2022,
			"Mileage": 15000,
			"Fuel Type": "Electric",
			"Transmission": "Automatic"
		}
	}'

# Batch prediction
curl -s -X POST http://127.0.0.1:8001/predict/batch \
	-H "Content-Type: application/json" \
	-d '{
		"records": [
			{
				"row_id": 1,
				"features": {
					"Brand": "Tesla",
					"Model": "Model 3",
					"Year": 2022,
					"Mileage": 15000,
					"Fuel Type": "Electric",
					"Transmission": "Automatic"
				}
			}
		]
	}'
```

Request body for single prediction:

```json
{
	"row_id": "optional-id",
	"features": {
		"Brand": "Tesla",
		"Model": "Model 3",
		"Year": 2022,
		"Mileage": 15000,
		"Fuel Type": "Electric",
		"Transmission": "Automatic"
	}
}
```

Response fields:
- `predicted_log_price`
- `predicted_price`
- `missing_features` (required columns not provided)
- `extra_features` (ignored inputs not in schema)

Batch request body:

```json
{
	"records": [
		{
			"row_id": 1,
			"features": {
				"Brand": "Tesla",
				"Model": "Model 3",
				"Year": 2022,
				"Mileage": 15000,
				"Fuel Type": "Electric",
				"Transmission": "Automatic"
			}
		}
	]
}
```

Sample `/schema` response:

```json
{
	"feature_columns": ["Brand", "Model", "Year", "Mileage", "Fuel Type", "Transmission"],
	"numeric_columns": ["Year", "Mileage"],
	"categorical_columns": ["Brand", "Model", "Fuel Type", "Transmission"],
	"feature_descriptions": {
		"Brand": "Vehicle brand",
		"Mileage": "Odometer reading"
	},
	"schema_source": "model"
}
```

Sample `/model/info` response:

```json
{
	"model_type": "Pipeline",
	"pipeline_steps": ["preprocess", "model"],
	"n_features_in": 24,
	"schema_source": "model"
}
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

## Notes and Assumptions

- The schema is derived from the trained model when available; otherwise it falls back to the data file header.
- The EDA indicates synthetic or randomized patterns in the source data, which can limit real-world generalization.
- Feature redundancy exists (for example `car_age` vs `age_squared` and CPI vs energy pressure), so consider feature selection if you retrain.

## Reproducibility

Notebooks:
- `Price Prediction EDA.ipynb` for feature engineering and enrichment.
- `Price Prediction Model Development.ipynb` for model training and evaluation.
