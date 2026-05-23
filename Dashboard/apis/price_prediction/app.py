"""
Price Prediction API

FastAPI service for predicting car prices from enriched feature inputs.

Usage:
    uvicorn apis.price_prediction.app:app --reload --port 8001
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app_config import (
    PRICE_ALT_DATA_PATH,
    PRICE_DATA_PATH,
    PRICE_FEATURE_DICT_PATH,
    PRICE_MODEL_PATH,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EV Price Prediction API",
    description="Predicts car prices from enriched feature inputs",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
FEATURE_COLUMNS: List[str] = []
NUMERIC_COLUMNS: List[str] = []
CATEGORICAL_COLUMNS: List[str] = []
FEATURE_DESCRIPTIONS: Dict[str, str] = {}
SCHEMA_SOURCE: Optional[str] = None


class PredictionRequest(BaseModel):
    row_id: Optional[Union[str, int]] = Field(default=None)
    features: Dict[str, Any] = Field(..., description="Raw feature inputs")


class PredictionRecord(BaseModel):
    row_id: Optional[Union[str, int]] = Field(default=None)
    features: Dict[str, Any] = Field(..., description="Raw feature inputs")


class PredictionResponse(BaseModel):
    row_id: Optional[Union[str, int]]
    predicted_log_price: float
    predicted_price: float
    missing_features: List[str]
    extra_features: List[str]


class BatchPredictionRequest(BaseModel):
    records: List[PredictionRecord]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    timestamp: datetime


class SchemaResponse(BaseModel):
    feature_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    feature_descriptions: Dict[str, str]
    schema_source: Optional[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: datetime
    feature_count: int


class ModelInfoResponse(BaseModel):
    model_type: str
    pipeline_steps: List[str]
    n_features_in: Optional[int]
    schema_source: Optional[str]


def _get_model_feature_names(model) -> Optional[List[str]]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        preprocess = model.named_steps.get("preprocess")
        if preprocess is not None and hasattr(preprocess, "feature_names_in_"):
            return list(preprocess.feature_names_in_)
    return None


def _load_feature_descriptions() -> Dict[str, str]:
    if not PRICE_FEATURE_DICT_PATH.exists():
        return {}
    try:
        df = pd.read_csv(PRICE_FEATURE_DICT_PATH)
        if "column" in df.columns:
            column_col = "column"
        elif "feature" in df.columns:
            column_col = "feature"
        else:
            return {}
        desc_col = "description" if "description" in df.columns else None
        if desc_col is None:
            return {}
        return (
            df[[column_col, desc_col]]
            .dropna(subset=[column_col])
            .set_index(column_col)[desc_col]
            .fillna("")
            .to_dict()
        )
    except Exception as exc:
        logger.warning("Feature dictionary load failed: %s", exc)
        return {}


def _load_data_schema() -> Tuple[Optional[List[str]], List[str], Optional[str]]:
    data_path = None
    for candidate in [PRICE_DATA_PATH, PRICE_ALT_DATA_PATH]:
        if candidate.exists():
            data_path = candidate
            break
    if data_path is None:
        return None, [], None

    df = pd.read_csv(data_path, nrows=500)
    drop_cols = [c for c in ["Price", "Log_Price", "Car ID"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    numeric_cols = (
        df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
        if feature_cols
        else []
    )
    return feature_cols, numeric_cols, str(data_path)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalize_features(
    features: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    missing = [c for c in FEATURE_COLUMNS if c not in features]
    extra = [c for c in features.keys() if c not in FEATURE_COLUMNS]
    normalized = {c: features.get(c, np.nan) for c in FEATURE_COLUMNS}
    return normalized, missing, extra


@app.on_event("startup")
async def startup_event() -> None:
    global MODEL, FEATURE_COLUMNS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, FEATURE_DESCRIPTIONS, SCHEMA_SOURCE

    try:
        MODEL = joblib.load(PRICE_MODEL_PATH)
        logger.info("Model loaded: %s", type(MODEL).__name__)
    except Exception as exc:
        logger.error("Failed to load model at %s: %s", PRICE_MODEL_PATH, exc)
        raise

    data_feature_cols, data_numeric_cols, data_source = _load_data_schema()
    model_feature_cols = _get_model_feature_names(MODEL)

    if model_feature_cols:
        FEATURE_COLUMNS = model_feature_cols
        SCHEMA_SOURCE = "model"
    elif data_feature_cols:
        FEATURE_COLUMNS = data_feature_cols
        SCHEMA_SOURCE = "data"
    else:
        raise RuntimeError("Unable to determine feature schema from model or data.")

    if data_numeric_cols:
        NUMERIC_COLUMNS = [c for c in data_numeric_cols if c in FEATURE_COLUMNS]
    else:
        NUMERIC_COLUMNS = []

    CATEGORICAL_COLUMNS = [c for c in FEATURE_COLUMNS if c not in NUMERIC_COLUMNS]

    if data_source and SCHEMA_SOURCE == "model":
        SCHEMA_SOURCE = f"model (data fallback at {data_source})"
    elif data_source:
        SCHEMA_SOURCE = f"data ({data_source})"

    FEATURE_DESCRIPTIONS = _load_feature_descriptions()
    logger.info("Feature schema loaded: %d features", len(FEATURE_COLUMNS))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=MODEL is not None,
        timestamp=datetime.utcnow(),
        feature_count=len(FEATURE_COLUMNS),
    )


@app.get("/schema", response_model=SchemaResponse)
async def schema() -> SchemaResponse:
    if not FEATURE_COLUMNS:
        raise HTTPException(status_code=503, detail="Schema not loaded yet.")
    return SchemaResponse(
        feature_columns=FEATURE_COLUMNS,
        numeric_columns=NUMERIC_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
        feature_descriptions=FEATURE_DESCRIPTIONS,
        schema_source=SCHEMA_SOURCE,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    steps = list(getattr(MODEL, "named_steps", {}).keys())
    n_features_in = getattr(MODEL, "n_features_in_", None)
    return ModelInfoResponse(
        model_type=type(MODEL).__name__,
        pipeline_steps=steps,
        n_features_in=n_features_in,
        schema_source=SCHEMA_SOURCE,
    )


def _predict(records: List[PredictionRecord]) -> List[PredictionResponse]:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not FEATURE_COLUMNS:
        raise HTTPException(status_code=503, detail="Schema not loaded yet.")

    normalized_rows: List[Dict[str, Any]] = []
    missing_list: List[List[str]] = []
    extra_list: List[List[str]] = []

    for record in records:
        normalized, missing, extra = _normalize_features(record.features)
        normalized_rows.append(normalized)
        missing_list.append(missing)
        extra_list.append(extra)

    df = pd.DataFrame(normalized_rows, columns=FEATURE_COLUMNS)
    df = _coerce_numeric(df)

    pred_log = MODEL.predict(df)
    pred_price = np.maximum(0, np.expm1(pred_log))

    responses: List[PredictionResponse] = []
    for idx, record in enumerate(records):
        responses.append(
            PredictionResponse(
                row_id=record.row_id,
                predicted_log_price=float(pred_log[idx]),
                predicted_price=float(pred_price[idx]),
                missing_features=missing_list[idx],
                extra_features=extra_list[idx],
            )
        )
    return responses


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    record = PredictionRecord(row_id=request.row_id, features=request.features)
    responses = _predict([record])
    return responses[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    predictions = _predict(request.records)
    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
        timestamp=datetime.utcnow(),
    )
