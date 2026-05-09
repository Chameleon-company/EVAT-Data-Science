from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

from price_app_config import PRICE_API_BASE_URL, PRICE_DATA_PATH, PRICE_ALT_DATA_PATH

st.set_page_config(page_title="EV Price Prediction", layout="wide")

st.title("EV Price Prediction Dashboard")
st.caption("Predict car prices using the trained regression model.")

API_TIMEOUT_SECONDS = 30
DATA_PREVIEW_ROWS = 5000
DEFAULT_FUEL_OPTIONS = ["Electric", "Petrol", "Diesel", "Hybrid"]
DEFAULT_MIN_YEAR = 1980

CONDITION_SCORES = {
    "new": 3,
    "like new": 3,
    "used": 2,
    "fair": 1,
    "old": 1,
}


def _resolve_data_path() -> Optional[str]:
    for candidate in [PRICE_DATA_PATH, PRICE_ALT_DATA_PATH]:
        if candidate.exists():
            return str(candidate)
    return None


def _to_jsonable(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


@st.cache_data(show_spinner=False)
def load_preview_data() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    data_path = _resolve_data_path()
    if data_path is None:
        return None, None
    try:
        df = pd.read_csv(data_path, nrows=DATA_PREVIEW_ROWS)
        if df.empty:
            return None, data_path
        return df, data_path
    except Exception:
        return None, data_path


@st.cache_data(show_spinner=False)
def load_schema(api_base_url: str) -> Dict[str, Any]:
    response = requests.get(f"{api_base_url}/schema", timeout=API_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _safe_float(value: Any) -> Optional[float]:
    try:
        if _is_missing(value):
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if _is_missing(value):
            return None
        return int(float(value))
    except Exception:
        return None


def _infer_default_values(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for col in feature_cols:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            value = series.dropna().median()
            defaults[col] = float(value) if not _is_missing(value) else None
        else:
            mode = series.dropna().mode()
            defaults[col] = mode.iloc[0] if not mode.empty else None
    return defaults


def _build_bin_helper(df: pd.DataFrame, value_col: str, bin_col: str):
    if value_col not in df.columns or bin_col not in df.columns:
        return None
    series = df[value_col].dropna()
    if series.empty:
        return None
    quantiles = series.quantile([0.0, 0.33, 0.66, 1.0]).values
    edges = np.unique(quantiles)
    if len(edges) < 4:
        return None
    bins = pd.cut(df[value_col], bins=edges, include_lowest=True)
    mapping: Dict[Any, Any] = {}
    for interval in bins.cat.categories:
        mask = bins == interval
        labels = df.loc[mask, bin_col].dropna().mode()
        if not labels.empty:
            mapping[interval] = labels.iloc[0]
    if not mapping:
        return None
    return edges, mapping


def _map_to_bin(value: Optional[float], helper) -> Optional[Any]:
    if helper is None or value is None:
        return None
    edges, mapping = helper
    if value < edges[0]:
        value = edges[0]
    if value > edges[-1]:
        value = edges[-1]
    interval = pd.cut([value], bins=edges, include_lowest=True)[0]
    return mapping.get(interval)


def _set_if_required(
    features: Dict[str, Any], required_columns: List[str], key: str, value: Any
) -> None:
    if key in features or key in required_columns:
        features[key] = value


def _derive_features(
    features: Dict[str, Any],
    required_columns: List[str],
    mileage_bin_helper,
    engine_bin_helper,
    mileage_high: Optional[float],
    mileage_very_high: Optional[float],
    reference_year: int,
    force_age: Optional[int] = None,
) -> Dict[str, Any]:
    updated = dict(features)

    year_value = _safe_int(updated.get("Year"))
    car_age = _safe_int(updated.get("car_age"))
    if force_age is not None:
        car_age = max(force_age, 0)
    elif car_age is None and year_value is not None:
        car_age = max(reference_year - year_value, 0)

    if car_age is not None:
        _set_if_required(updated, required_columns, "car_age", car_age)
        _set_if_required(updated, required_columns, "age_squared", car_age**2)
        _set_if_required(updated, required_columns, "car_age_sq", car_age**2)

    mileage = _safe_float(updated.get("Mileage"))
    if mileage is not None:
        _set_if_required(updated, required_columns, "log_mileage", np.log1p(mileage))
    if mileage is not None and car_age is not None:
        per_year = mileage if car_age <= 0 else mileage / max(car_age, 1)
        _set_if_required(updated, required_columns, "mileage_per_year", per_year)

    engine_size = _safe_float(updated.get("Engine Size"))
    if engine_size is not None and mileage is not None:
        _set_if_required(
            updated,
            required_columns,
            "engine_size_per_mileage",
            engine_size / max(mileage, 1.0),
        )

    condition = updated.get("Condition")
    if condition is not None:
        score = CONDITION_SCORES.get(str(condition).strip().lower(), None)
        if score is not None:
            _set_if_required(updated, required_columns, "condition_score", score)

    if mileage is not None:
        mileage_bin = _map_to_bin(mileage, mileage_bin_helper)
        if mileage_bin is not None:
            _set_if_required(updated, required_columns, "mileage_bin", mileage_bin)
            _set_if_required(updated, required_columns, "mileage_band", mileage_bin)

    if engine_size is not None:
        engine_bin = _map_to_bin(engine_size, engine_bin_helper)
        if engine_bin is not None:
            _set_if_required(updated, required_columns, "engine_size_bin", engine_bin)

    mileage_bin_value = updated.get("mileage_bin") or updated.get("mileage_band")
    if condition is not None and mileage_bin_value is not None:
        _set_if_required(
            updated,
            required_columns,
            "condition_mileage_bin",
            f"{condition}_{mileage_bin_value}",
        )

    brand = updated.get("Brand")
    model = updated.get("Model")
    if brand and model:
        _set_if_required(updated, required_columns, "brand_model", f"{brand}_{model}")

    fuel_type = updated.get("Fuel Type")
    transmission = updated.get("Transmission")
    if fuel_type and transmission:
        _set_if_required(
            updated,
            required_columns,
            "fuel_transmission",
            f"{fuel_type}_{transmission}",
        )

    if mileage is not None and condition is not None:
        if mileage_high is not None:
            high_flag = int(str(condition).strip().lower() == "new" and mileage > mileage_high)
            _set_if_required(updated, required_columns, "new_high_mileage_flag", high_flag)
        if mileage_very_high is not None:
            very_high_flag = int(
                str(condition).strip().lower() == "new" and mileage > mileage_very_high
            )
            _set_if_required(
                updated, required_columns, "very_new_very_high_mileage_flag", very_high_flag
            )

    if fuel_type is not None:
        is_ev = str(fuel_type).strip().lower() == "electric"
        if engine_size is not None:
            mismatch = int(is_ev and engine_size > 0.1)
            _set_if_required(updated, required_columns, "ev_engine_size_mismatch_flag", mismatch)
        if transmission is not None:
            manual_ev = int(is_ev and str(transmission).strip().lower() == "manual")
            _set_if_required(updated, required_columns, "manual_ev_flag", manual_ev)

    return updated


def _filter_to_schema(features: Dict[str, Any], required_columns: List[str]) -> Dict[str, Any]:
    if not required_columns:
        return features
    return {key: features.get(key) for key in required_columns}


def _call_predict(api_base_url: str, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    response = requests.post(
        f"{api_base_url}/predict",
        json={"features": features},
        timeout=API_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _call_predict_batch(
    api_base_url: str, records: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    response = requests.post(
        f"{api_base_url}/predict/batch",
        json={"records": records},
        timeout=API_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _resolve_compare_fuel(selected_fuel: str, fuel_options: List[str]) -> Optional[str]:
    if not selected_fuel:
        return None
    selected = str(selected_fuel).strip().lower()
    options_lower = {opt.lower(): opt for opt in fuel_options}
    if selected == "electric":
        for candidate in ["petrol", "diesel", "hybrid"]:
            if candidate in options_lower:
                return options_lower[candidate]
        return "Petrol"
    if "electric" in options_lower:
        return options_lower["electric"]
    return None


def _get_price_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if "Price" in df.columns:
        return pd.to_numeric(df["Price"], errors="coerce")
    if "Log_Price" in df.columns:
        return np.expm1(pd.to_numeric(df["Log_Price"], errors="coerce"))
    return None


def _estimate_selected_price(
    df: pd.DataFrame, brand: str, model: str, year_value: int, fuel: str
) -> Optional[float]:
    price_series = _get_price_series(df)
    if price_series is None:
        return None

    def _apply_filter(frame: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
        if column not in frame.columns or _is_missing(value) or value == "":
            return frame
        return frame.loc[frame[column] == value]

    rows = df
    rows = _apply_filter(rows, "Brand", brand)
    rows = _apply_filter(rows, "Model", model)
    rows = _apply_filter(rows, "Year", year_value)
    rows = _apply_filter(rows, "Fuel Type", fuel)

    if rows.empty:
        rows = df
        rows = _apply_filter(rows, "Brand", brand)
        rows = _apply_filter(rows, "Model", model)
        rows = _apply_filter(rows, "Fuel Type", fuel)

    if rows.empty:
        rows = df
        rows = _apply_filter(rows, "Brand", brand)
        rows = _apply_filter(rows, "Model", model)

    if rows.empty:
        rows = df
        rows = _apply_filter(rows, "Fuel Type", fuel)

    if rows.empty:
        rows = df

    prices = price_series.loc[rows.index].dropna()
    if prices.empty:
        return None
    return float(prices.median())


def _find_ev_equivalent(
    df: pd.DataFrame, brand: str, model: str, year_value: int, fuel: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if df is None or "Fuel Type" not in df.columns:
        return None, None

    price_series = _get_price_series(df)
    if price_series is None:
        return None, None

    ev_mask = df["Fuel Type"].astype(str).str.strip().str.lower() == "electric"
    ev_rows = df.loc[ev_mask]
    if ev_rows.empty:
        return None, None

    target_price = _estimate_selected_price(df, brand, model, year_value, fuel)
    if target_price is None:
        return None, None

    ev_prices = price_series.loc[ev_rows.index].dropna()
    if ev_prices.empty:
        return None, None

    quantiles = ev_prices.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).values
    edges = np.unique(quantiles)

    if len(edges) < 2:
        closest_idx = (ev_prices - target_price).abs().idxmin()
        match = ev_rows.loc[closest_idx]
    else:
        tier_index = int(np.searchsorted(edges, target_price, side="right") - 1)
        tier_index = max(0, min(tier_index, len(edges) - 2))
        lower = edges[tier_index]
        upper = edges[tier_index + 1]
        tier_prices = ev_prices[(ev_prices >= lower) & (ev_prices <= upper)]
        if tier_prices.empty:
            tier_prices = ev_prices
        closest_idx = (tier_prices - target_price).abs().idxmin()
        match = ev_rows.loc[closest_idx]

    match_payload = {
        "Brand": match.get("Brand"),
        "Model": match.get("Model"),
        "Year": _safe_int(match.get("Year")),
        "Fuel Type": match.get("Fuel Type"),
        "Mileage": _safe_float(match.get("Mileage")),
    }
    label = f"Equivalent EV: {match_payload.get('Brand', '')} {match_payload.get('Model', '')}".strip()
    return match_payload, label


st.sidebar.header("Settings")
api_base_url = st.sidebar.text_input("API base URL", PRICE_API_BASE_URL)

if st.sidebar.button("Check API health"):
    try:
        resp = requests.get(f"{api_base_url}/health", timeout=API_TIMEOUT_SECONDS)
        resp.raise_for_status()
        st.sidebar.success(resp.json())
    except requests.RequestException as exc:
        st.sidebar.error("Health check failed. Check API connection.")

schema_data: Dict[str, Any] = {}
if "schema_data" not in st.session_state:
    try:
        schema_data = load_schema(api_base_url)
        st.session_state["schema_data"] = schema_data
    except requests.RequestException:
        schema_data = {}

if st.sidebar.button("Reload schema"):
    try:
        schema_data = load_schema(api_base_url)
        st.sidebar.success("Schema loaded")
        st.session_state["schema_data"] = schema_data
    except requests.RequestException:
        st.sidebar.error("Schema load failed. Check API connection.")

if "schema_data" in st.session_state:
    schema_data = st.session_state["schema_data"]

preview_df, preview_path = load_preview_data()

feature_cols: List[str] = []
defaults: Dict[str, Any] = {}
sample_row: Dict[str, Any] = {}
required_columns: List[str] = []
mileage_bin_helper = None
engine_bin_helper = None
mileage_high = None
mileage_very_high = None

if preview_df is not None:
    drop_cols = [c for c in ["Price", "Log_Price", "Car ID"] if c in preview_df.columns]
    feature_cols = [c for c in preview_df.columns if c not in drop_cols]
    defaults = _infer_default_values(preview_df, feature_cols)
    sample_row = (
        preview_df[feature_cols].sample(1, random_state=42).iloc[0].to_dict()
    )
    mileage_bin_helper = _build_bin_helper(preview_df, "Mileage", "mileage_bin")
    if mileage_bin_helper is None:
        mileage_bin_helper = _build_bin_helper(preview_df, "Mileage", "mileage_band")
    engine_bin_helper = _build_bin_helper(preview_df, "Engine Size", "engine_size_bin")
    if "Mileage" in preview_df.columns:
        mileage_series = preview_df["Mileage"].dropna()
        if not mileage_series.empty:
            mileage_high = float(mileage_series.quantile(0.75))
            mileage_very_high = float(mileage_series.quantile(0.9))

if schema_data:
    required_columns = schema_data.get("feature_columns", [])
elif feature_cols:
    required_columns = feature_cols

brands = (
    sorted(preview_df["Brand"].dropna().unique().tolist())
    if preview_df is not None and "Brand" in preview_df.columns
    else []
)
fuel_options = (
    sorted(preview_df["Fuel Type"].dropna().unique().tolist())
    if preview_df is not None and "Fuel Type" in preview_df.columns
    else DEFAULT_FUEL_OPTIONS
)

default_brand = sample_row.get("Brand") or (brands[0] if brands else "")
default_model = sample_row.get("Model")
default_year = _safe_int(sample_row.get("Year"))
default_fuel = sample_row.get("Fuel Type") or (fuel_options[0] if fuel_options else "")
default_mileage = _safe_float(sample_row.get("Mileage"))
if default_year is None:
    default_year = datetime.now().year
if default_mileage is None:
    default_mileage = _safe_float(defaults.get("Mileage")) or 0.0

st.subheader("Vehicle selection")
col_brand, col_model, col_year, col_fuel, col_mileage = st.columns(5)

with col_brand:
    if brands:
        brand_index = brands.index(default_brand) if default_brand in brands else 0
        brand = st.selectbox("Brand", brands, index=brand_index)
    else:
        brand = st.text_input("Brand", value=str(default_brand or ""))

with col_model:
    if preview_df is not None and "Model" in preview_df.columns:
        if brand and "Brand" in preview_df.columns:
            model_options = (
                preview_df.loc[preview_df["Brand"] == brand, "Model"]
                .dropna()
                .unique()
                .tolist()
            )
        else:
            model_options = preview_df["Model"].dropna().unique().tolist()
        model_options = sorted(model_options)
        if model_options:
            if default_model in model_options:
                model_index = model_options.index(default_model)
            else:
                model_index = 0
            model = st.selectbox("Model", model_options, index=model_index)
        else:
            model = st.text_input("Model", value=str(default_model or ""))
    else:
        model = st.text_input("Model", value=str(default_model or ""))

with col_year:
    year = st.number_input(
        "Year",
        min_value=DEFAULT_MIN_YEAR,
        max_value=datetime.now().year + 1,
        value=int(default_year),
        step=1,
    )

with col_fuel:
    fuel_index = fuel_options.index(default_fuel) if default_fuel in fuel_options else 0
    fuel_type = st.selectbox("Fuel Type", fuel_options, index=fuel_index)

with col_mileage:
    mileage = st.number_input(
        "Mileage (km)", min_value=0.0, value=float(default_mileage), step=100.0
    )

auto_derive = st.checkbox("Auto-derive dependent features", value=True)

other_feature_cols = [
    col
    for col in (required_columns or feature_cols)
    if col not in {"Brand", "Model", "Year", "Fuel Type", "Mileage"}
]

if "other_defaults" not in st.session_state:
    other_defaults = {
        key: _to_jsonable(defaults.get(key)) for key in other_feature_cols
    }
    st.session_state["other_defaults"] = other_defaults

other_defaults = st.session_state.get("other_defaults", {})

if other_feature_cols:
    editor_df = pd.DataFrame(
        {
            "Feature": other_feature_cols,
            "Value": [other_defaults.get(col) for col in other_feature_cols],
        }
    )
    with st.expander("Other variables"):
        st.caption(
            "Update any defaults as needed; derived fields update automatically when enabled."
        )
        edited_df = st.data_editor(
            editor_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "Feature": st.column_config.TextColumn(disabled=True),
            },
        )
        other_overrides = {
            row["Feature"]: row["Value"] for _, row in edited_df.iterrows()
        }
        st.session_state["other_defaults"] = other_overrides
else:
    other_overrides = {}


def _build_feature_payload(
    base_year: int,
    force_age: Optional[int] = None,
    override_fuel: Optional[str] = None,
    override_fields: Optional[Dict[str, Any]] = None,
    override_mileage: Optional[float] = None,
    force_derive: bool = False,
) -> Dict[str, Any]:
    payload = dict(defaults)
    payload.update(other_overrides)
    payload.update(
        {
            "Brand": brand,
            "Model": model,
            "Year": int(year),
            "Fuel Type": fuel_type if override_fuel is None else override_fuel,
            "Mileage": float(mileage),
        }
    )

    if override_fields:
        payload.update(override_fields)

    if override_mileage is not None:
        payload["Mileage"] = float(override_mileage)

    if auto_derive or force_derive:
        payload = _derive_features(
            payload,
            required_columns,
            mileage_bin_helper,
            engine_bin_helper,
            mileage_high,
            mileage_very_high,
            reference_year=base_year,
            force_age=force_age,
        )

    return _filter_to_schema(payload, required_columns)

st.subheader("EV vs non-EV price projection")

forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
with forecast_col1:
    horizon_years = st.slider(
        "Forecast horizon (years)", min_value=5, max_value=7, value=6
    )
with forecast_col2:
    compare_enabled = st.checkbox("Compare EV vs non-EV", value=True)
with forecast_col3:
    mileage_growth_pct = st.number_input(
        "Annual mileage growth (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.5
    )

forecast_button = st.button("Generate forecast")

if forecast_button:
    base_year = datetime.now().year
    time_sensitive_cols = {
        "car_age",
        "car_age_sq",
        "age_squared",
        "Year",
        "Mileage",
        "log_mileage",
        "mileage_per_year",
    }
    if not time_sensitive_cols.intersection(required_columns):
        st.warning(
            "Model schema has no time-varying inputs, so forecasts may be flat year-on-year."
        )
    base_age = _safe_int(defaults.get("car_age"))
    if base_age is None:
        base_age = _safe_int(other_overrides.get("car_age"))
    if base_age is None:
        base_age = max(base_year - int(year), 0)

    records: List[Dict[str, Any]] = []
    years = list(range(base_year, base_year + horizon_years))
    base_mileage = float(mileage)
    growth_rate = float(mileage_growth_pct) / 100.0
    use_year_for_age = "car_age" not in required_columns and "Year" in required_columns
    base_vehicle_year = int(year)

    for offset, year_value in enumerate(years):
        mileage_year = base_mileage * ((1.0 + growth_rate) ** offset)
        override_year = None
        if use_year_for_age:
            override_year = max(DEFAULT_MIN_YEAR, base_vehicle_year - offset)
        payload = _build_feature_payload(
            base_year=year_value,
            force_age=base_age + offset,
            override_fields={"Year": override_year} if override_year is not None else None,
            override_mileage=mileage_year,
            force_derive=True,
        )
        records.append({"row_id": f"selected_{year_value}", "features": payload})

    compare_fuel = None
    compare_label = None
    compare_override = None
    compare_base_age = base_age
    compare_base_mileage = base_mileage
    compare_vehicle_year = base_vehicle_year
    if compare_enabled:
        selected_is_ev = str(fuel_type).strip().lower() == "electric"
        if selected_is_ev:
            compare_fuel = _resolve_compare_fuel(fuel_type, fuel_options)
            if compare_fuel:
                compare_label = "Equivalent non-EV"
        else:
            if preview_df is None:
                st.info("EV comparison unavailable: preview data not loaded.")
            else:
                compare_override, compare_label = _find_ev_equivalent(
                    preview_df, brand, model, int(year), fuel_type
                )
                if compare_override and compare_override.get("Fuel Type"):
                    compare_fuel = compare_override.get("Fuel Type")
                    ev_year = _safe_int(compare_override.get("Year"))
                    if ev_year is not None:
                        compare_base_age = max(base_year - ev_year, 0)
                        compare_vehicle_year = ev_year
                    ev_mileage = _safe_float(compare_override.get("Mileage"))
                    if ev_mileage is not None:
                        compare_base_mileage = ev_mileage
                    ev_brand = compare_override.get("Brand") or ""
                    ev_model = compare_override.get("Model") or ""
                    ev_year_label = compare_override.get("Year")
                    ev_desc = f"{ev_brand} {ev_model}".strip()
                    if ev_year_label:
                        ev_desc = f"{ev_desc} ({ev_year_label})".strip()
                    if ev_desc:
                        st.success(f"Equivalent EV match: {ev_desc}")

        if compare_fuel:
            for offset, year_value in enumerate(years):
                mileage_year = compare_base_mileage * ((1.0 + growth_rate) ** offset)
                compare_override_fields = dict(compare_override or {})
                if use_year_for_age:
                    compare_override_fields["Year"] = max(
                        DEFAULT_MIN_YEAR, compare_vehicle_year - offset
                    )
                payload = _build_feature_payload(
                    base_year=year_value,
                    force_age=compare_base_age + offset,
                    override_fuel=compare_fuel,
                    override_fields=compare_override_fields or None,
                    override_mileage=mileage_year,
                    force_derive=True,
                )
                records.append({"row_id": f"compare_{year_value}", "features": payload})

    try:
        response = _call_predict_batch(api_base_url, records)
        predictions = response.get("predictions", [])
        rows = []
        for item in predictions:
            row_id = item.get("row_id", "")
            if row_id.startswith("selected_"):
                variant = f"Selected ({fuel_type})"
                year_value = int(row_id.replace("selected_", ""))
            elif row_id.startswith("compare_"):
                variant = (
                    f"{compare_label} ({compare_fuel})"
                    if compare_label
                    else "Comparison"
                )
                year_value = int(row_id.replace("compare_", ""))
            else:
                variant = "Selected"
                year_value = base_year
            rows.append(
                {
                    "Year": year_value,
                    "Variant": variant,
                    "Predicted Price": float(item.get("predicted_price", 0.0)),
                }
            )

        import altair as alt
        forecast_df = pd.DataFrame(rows)
        if not forecast_df.empty:
            chart = alt.Chart(forecast_df).mark_line(point=True).encode(
                x=alt.X('Year:O', title='Year'),
                y=alt.Y('Predicted Price:Q', scale=alt.Scale(zero=False)),
                color='Variant:N',
                tooltip=['Year', 'Variant', 'Predicted Price']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
            st.caption(
                f"Age increases yearly; mileage increases {mileage_growth_pct:.1f}% per year."
            )
            st.dataframe(
                forecast_df.sort_values(["Year", "Variant"]),
                use_container_width=True,
                hide_index=True,
            )
    except requests.RequestException:
        st.sidebar.error("Forecast failed. Check API connection.")

if schema_data:
    with st.expander("Feature schema"):
        st.write(schema_data)
