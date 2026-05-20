import os
from pathlib import Path

PRICE_ROOT = Path(__file__).resolve().parent


def _resolve_path(env_var: str, fallback_relative_path: Path) -> Path:
    configured = os.getenv(env_var)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = PRICE_ROOT / path
        return path.resolve()

    return (PRICE_ROOT / fallback_relative_path).resolve()


PRICE_API_HOST = os.getenv("PRICE_API_HOST", "127.0.0.1")
PRICE_API_PORT = int(os.getenv("PRICE_API_PORT", "8001"))
PRICE_DASHBOARD_PORT = int(os.getenv("PRICE_DASHBOARD_PORT", "8502"))
PRICE_API_BASE_URL = os.getenv(
    "PRICE_API_BASE_URL", f"http://{PRICE_API_HOST}:{PRICE_API_PORT}"
).rstrip("/")

PRICE_MODEL_PATH = _resolve_path(
    "PRICE_MODEL_PATH", Path("artifacts") / "price_best_model_latest.joblib"
)
PRICE_DATA_PATH = _resolve_path(
    "PRICE_DATA_PATH", Path("artifacts") / "car_price_enriched_latest.csv"
)
PRICE_ALT_DATA_PATH = _resolve_path(
    "PRICE_ALT_DATA_PATH", Path("car_price_prediction_enriched_features.csv")
)
PRICE_FEATURE_DICT_PATH = _resolve_path(
    "PRICE_FEATURE_DICT_PATH", Path("artifacts") / "feature_dictionary.csv"
)
