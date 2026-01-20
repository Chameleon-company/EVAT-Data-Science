import pandas as pd
import numpy as np
import os
from pathlib import Path

# Define paths relative to the current file
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = CURRENT_DIR / ".." / "data" / "raw"
PROCESSED_PATH = CURRENT_DIR / ".." / "data" / "processed"

def load_station_data():
    """Load and clean charger station data."""
    df = pd.read_csv(RAW_PATH / "open_charge_map_au.csv")
    df_scores = pd.read_csv(RAW_PATH / "charger_accessibility_scores.csv")
    df_underserved = pd.read_csv(RAW_PATH / "underserved_chargers.csv")

    # Rename columns for consistency
    df.rename(columns={
        "ID": "station_id", 
        "Title": "title",
        "Latitude": "latitude", 
        "Longitude": "longitude",
        "City": "city", 
        "Connectors": "num_connectors",
        "WheelchairAccess": "wheelchair_access"
    }, inplace=True)

    df_scores.rename(columns={"ID": "station_id"}, inplace=True)
    df_underserved.rename(columns={"ID": "station_id"}, inplace=True)

    # Merge station info with scores
    merged = df.merge(df_scores, on="station_id", suffixes=("", "_score"))
    merged["underserved"] = merged["station_id"].isin(df_underserved["station_id"])

    return merged

def load_sessions_data():
    """Load and clean EV charging session data."""
    df = pd.read_csv(RAW_PATH / "station_data_dataverse.csv")
    
    # Fix invalid years in timestamps
    df["created"] = df["created"].astype(str).str.replace("0014", "2014").str.replace("0015", "2015")
    df["ended"] = df["ended"].astype(str).str.replace("0014", "2014").str.replace("0015", "2015")

    # Convert to datetime
    df["start_datetime"] = pd.to_datetime(df["created"], errors="coerce")
    df["end_datetime"] = pd.to_datetime(df["ended"], errors="coerce")

    # Compute duration
    df["duration_hrs"] = (df["end_datetime"] - df["start_datetime"]).dt.total_seconds() / 3600

    # Rename columns
    df.rename(columns={
        "stationId": "station_id",
        "kwhTotal": "kwh_total",
        "dollars": "cost_dollars",
        "distance": "distance_km"
    }, inplace=True)

    return df

def load_weather_data():
    """Load and clean weather dataset."""
    df = pd.read_csv(RAW_PATH / "weather_data.csv")
    df.rename(columns={
        "STATION": "station_code",
        "NAME": "station_name",
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "ELEVATION": "elevation_m",
        "DATE": "date",
        "PRCP": "precipitation_mm",
        "TAVG": "temp_avg_c",
        "TMAX": "temp_max_c",
        "TMIN": "temp_min_c"
    }, inplace=True)

    # Parse date and convert numerics
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    numeric_cols = ["precipitation_mm", "temp_avg_c", "temp_max_c", "temp_min_c"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Drop attribute columns if present
    df = df.drop(columns=[col for col in df.columns if "ATTRIBUTES" in col], errors='ignore')
    return df

def merge_charger_sessions(charger_df, session_df):
    """Merge session data with charger station info."""
    if charger_df is None or session_df is None:
        raise ValueError("Input dataframes cannot be None")
    return pd.merge(session_df, charger_df, on="station_id", how="left")

def clean_and_merge_data():
    """Complete pipeline: load, clean, merge, and save processed datasets."""
    weather_df = load_weather_data()
    charger_df = load_station_data()
    session_df = load_sessions_data()

    merged_charger_sessions_df = merge_charger_sessions(charger_df, session_df)

    # Save processed files
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    weather_df.to_csv(PROCESSED_PATH / "weather_data_clean.csv", index=False)
    charger_df.to_csv(PROCESSED_PATH / "stations_info_clean.csv", index=False)
    session_df.to_csv(PROCESSED_PATH / "charging_sessions_clean.csv", index=False)
    merged_charger_sessions_df.to_csv(PROCESSED_PATH / "sessions_with_stations.csv", index=False)

    return weather_df, charger_df, session_df, merged_charger_sessions_df

# if __name__ == "__main__":
#     clean_and_merge_data()
