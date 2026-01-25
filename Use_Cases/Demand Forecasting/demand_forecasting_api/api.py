# ==============================================================================
# EV CHARGING DEMAND PREDICTION API
# ==============================================================================

# ------------------------------------------------------------------------------
# SECTION 1: IMPORTS
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, date, timedelta
import holidays

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import uvicorn

# ------------------------------------------------------------------------------
# SECTION 2: LOAD ASSETS AT STARTUP
# ------------------------------------------------------------------------------
print("Loading application assets...")

try:
    # Load the trained LightGBM model
    model = joblib.load("ev_demand_model.pkl")
    print("✓ Model loaded")

    # Load the postcode baseline data
    postcode_baseline = pd.read_csv("postcode_baseline.csv")
    postcode_baseline['Postcode'] = postcode_baseline['Postcode'].astype(str)
    print("✓ Postcode baseline loaded")

    # Load the postcode coordinates as a dictionary
    coords_df = pd.read_csv("postcode_coords.csv")
    coords_df['Postcode'] = coords_df['Postcode'].astype(str)
    postcode_coords = {
        row['Postcode']: (row['lat'], row['lon']) 
        for _, row in coords_df.iterrows()
    }
    print("✓ Postcode coordinates loaded")

    # Load the feature columns
    with open("feature_columns.txt", "r") as f:
        feature_columns = f.read().strip().split(",")
    print(f"✓ Feature columns loaded: {feature_columns}")

    # Initialize the Australian holidays checker
    au_holidays = holidays.AU()
    print("✓ Holiday calendar initialized")

    print("All assets loaded successfully!\n")

except FileNotFoundError as e:
    print(f"FATAL ERROR: Missing required file - {e}")
    print("Please ensure all asset files are in the same directory as api.py")
    raise SystemExit(1)

# ------------------------------------------------------------------------------
# SECTION 3: HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def get_weather_forecast(lat: float, lon: float, target_date: date) -> float | None:
    """
    Fetches the forecasted mean temperature from Open-Meteo API.
    Returns None if the forecast cannot be retrieved.
    """
    days_ahead = (target_date - date.today()).days + 1
    
    if days_ahead < 1:
        raise ValueError("Cannot fetch forecast for past dates.")
    if days_ahead > 16:
        raise ValueError("Can only fetch forecasts for the next 16 days.")

    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_mean",
        "forecast_days": days_ahead,
        "timezone": "Australia/Sydney"
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        forecast_df = pd.DataFrame(data['daily'])
        forecast_df['time'] = pd.to_datetime(forecast_df['time']).dt.date

        temp = forecast_df[forecast_df['time'] == target_date]['temperature_2m_mean'].iloc[0]
        return float(temp)

    except requests.exceptions.RequestException as e:
        print(f"⚠ Weather API request failed: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"⚠ Failed to parse weather data: {e}")
        return None


def create_prediction_features(postcode: str, target_date: date) -> pd.DataFrame:
    """
    Creates a feature DataFrame for a single prediction.
    """
    # Look up baseline info for the postcode
    baseline_info = postcode_baseline[postcode_baseline['Postcode'] == postcode]
    if baseline_info.empty:
        raise ValueError(f"Postcode '{postcode}' not found in baseline data.")

    baseline_kwh = baseline_info['baseline_daily_kwh'].iloc[0]
    state = baseline_info['State'].iloc[0]

    # Look up coordinates for weather forecast
    if postcode not in postcode_coords:
        raise ValueError(f"Coordinates for postcode '{postcode}' not found.")
    lat, lon = postcode_coords[postcode]

    # Get weather forecast
    temperature = get_weather_forecast(lat, lon, target_date)
    if temperature is None:
        temperature = 20.0  # Default fallback temperature
        print(f"  ⚠ Using default temperature of {temperature}°C for postcode {postcode}")

    # Create date-based features
    target_datetime = datetime.combine(target_date, datetime.min.time())
    day_of_week = target_datetime.weekday()

    # Build the feature dictionary
    feature_dict = {
        'Postcode': postcode,
        'State': state,
        'baseline_daily_kwh': baseline_kwh,
        'day_of_week': day_of_week,
        'is_weekend': 1 if day_of_week >= 5 else 0,
        'month': target_date.month,
        'day_of_year': target_datetime.timetuple().tm_yday,
        'quarter': (target_date.month - 1) // 3 + 1,
        'year': target_date.year,
        'is_holiday': 1 if target_date in au_holidays else 0,
        'temperature': temperature
    }

    # Create DataFrame and set correct dtypes
    pred_df = pd.DataFrame([feature_dict])
    pred_df['Postcode'] = pred_df['Postcode'].astype('category')
    pred_df['State'] = pred_df['State'].astype('category')

    # Return only the columns the model expects, in the correct order
    return pred_df[feature_columns]


def predict_demand(postcode: str, target_date: date) -> dict:
    """
    Main prediction function that orchestrates the entire prediction pipeline.
    """
    try:
        # Create features
        X_pred = create_prediction_features(postcode, target_date)

        # Make prediction
        prediction = model.predict(X_pred)

        return {
            "postcode": postcode,
            "date": target_date.isoformat(),
            "predicted_demand_kwh": round(float(prediction[0]), 2),
            "status": "success"
        }

    except ValueError as e:
        return {
            "postcode": postcode,
            "date": target_date.isoformat(),
            "error": str(e),
            "status": "error"
        }
    except Exception as e:
        return {
            "postcode": postcode,
            "date": target_date.isoformat(),
            "error": f"Unexpected error: {str(e)}",
            "status": "error"
        }

# ------------------------------------------------------------------------------
# SECTION 4: FASTAPI APPLICATION
# ------------------------------------------------------------------------------

app = FastAPI(
    title="EV Charging Demand Prediction API",
    description="Predicts daily EV charging demand (kWh) for Australian postcodes.",
    version="1.0.0"
)

# Define the request body schema
class PredictionRequest(BaseModel):
    postcode: str
    date: date  # Expects "YYYY-MM-DD" format

    @field_validator('postcode')
    @classmethod
    def validate_postcode(cls, v):
        # Remove any whitespace and ensure it's a string
        return str(v).strip()

# Define the response schema
class PredictionResponse(BaseModel):
    postcode: str
    date: str
    predicted_demand_kwh: float
    status: str

class ErrorResponse(BaseModel):
    postcode: str
    date: str
    error: str
    status: str


# --- API Endpoints ---

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "message": "EV Charging Demand Prediction API is running!",
        "docs": "Visit /docs for interactive API documentation"
    }


@app.post("/predict", response_model=PredictionResponse)
def handle_prediction(request: PredictionRequest):
    """
    Predict EV charging demand for a given postcode and date.
    
    - **postcode**: Australian postcode (e.g., "2000" for Sydney)
    - **date**: Target date in YYYY-MM-DD format (must be within next 16 days)
    """
    result = predict_demand(request.postcode, request.date)

    if result['status'] == 'error':
        raise HTTPException(status_code=400, detail=result['error'])

    return result


@app.get("/postcodes")
def list_postcodes():
    """Returns a list of all valid postcodes the API can predict for."""
    valid_postcodes = postcode_baseline['Postcode'].tolist()
    return {
        "count": len(valid_postcodes),
        "postcodes": valid_postcodes
    }


# ------------------------------------------------------------------------------
# SECTION 5: RUN THE SERVER
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting the EV Demand Prediction API server...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)