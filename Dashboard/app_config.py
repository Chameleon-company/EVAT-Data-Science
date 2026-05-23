import os
from pathlib import Path

from dotenv import load_dotenv

DASHBOARD_ROOT = Path(__file__).resolve().parent
load_dotenv(DASHBOARD_ROOT / ".env")


def _resolve_path(env_var: str, fallback_relative_path: Path) -> Path:
    configured = os.getenv(env_var)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = DASHBOARD_ROOT / path
        return path.resolve()

    return (DASHBOARD_ROOT / fallback_relative_path).resolve()


API_HOST = os.getenv("EVAT_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("EVAT_API_PORT", "8000"))
DASHBOARD_PORT = int(os.getenv("EVAT_DASHBOARD_PORT", "8501"))
API_BASE_URL = os.getenv("EVAT_API_BASE_URL", f"http://{API_HOST}:{API_PORT}").rstrip("/")

PRICE_API_HOST = os.getenv("PRICE_API_HOST", "127.0.0.1")
PRICE_API_PORT = int(os.getenv("PRICE_API_PORT", "8001"))
PRICE_DASHBOARD_PORT = int(os.getenv("PRICE_DASHBOARD_PORT", "8502"))
PRICE_API_BASE_URL = os.getenv(
    "PRICE_API_BASE_URL", f"http://{PRICE_API_HOST}:{PRICE_API_PORT}"
).rstrip("/")

STATIONS_DATA_PATH = _resolve_path("EVAT_STATIONS_DATA_PATH", Path("data") / "EVAT.chargers.csv")
CONGESTION_MODEL_PATH = _resolve_path(
    "EVAT_CONGESTION_MODEL_PATH", Path("models") / "congestion_prediction" / "random_forest_model.pkl"
)
CONGESTION_TRAIN_DATA_PATH = _resolve_path(
    "EVAT_CONGESTION_TRAIN_DATA_PATH", Path("data") / "congestion_prediction" / "train_exogenous_3h.csv"
)

PRICE_MODEL_PATH = _resolve_path(
    "PRICE_MODEL_PATH",
    Path("..")
    / "Use_Cases"
    / "Price Prediction"
    / "artifacts"
    / "price_best_model_latest.joblib",
)
PRICE_DATA_PATH = _resolve_path(
    "PRICE_DATA_PATH",
    Path("..")
    / "Use_Cases"
    / "Price Prediction"
    / "artifacts"
    / "car_price_enriched_latest.csv",
)
PRICE_ALT_DATA_PATH = _resolve_path(
    "PRICE_ALT_DATA_PATH",
    Path("..") / "Use_Cases" / "Price Prediction" / "car_price_prediction_enriched_features.csv",
)
PRICE_FEATURE_DICT_PATH = _resolve_path(
    "PRICE_FEATURE_DICT_PATH",
    Path("..") / "Use_Cases" / "Price Prediction" / "artifacts" / "feature_dictionary.csv",
)
