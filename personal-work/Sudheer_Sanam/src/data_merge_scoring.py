import pandas as pd
import numpy as np
import os
from pathlib import Path
from geopy.distance import geodesic

CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = CURRENT_DIR / ".." / "data" / "processed"
INTERIM_PATH = CURRENT_DIR / ".." / "data" / "interim"

def load_clean_data():
    weather = pd.read_csv(PROCESSED_PATH / "weather_data_clean.csv")
    stations = pd.read_csv(PROCESSED_PATH / "stations_info_clean.csv")
    sessions = pd.read_csv(PROCESSED_PATH / "charging_sessions_clean.csv")
    merged_sessions = pd.read_csv(PROCESSED_PATH / "sessions_with_stations.csv")
    return weather, stations, sessions, merged_sessions

def find_nearest_station(lat, lon, weather_df):
    # Skip if coordinates are missing or invalid
    if pd.isna(lat) or pd.isna(lon):
        return np.nan

    valid_weather = weather_df.dropna(subset=["latitude", "longitude"]).copy()
    if valid_weather.empty:
        return np.nan

    valid_weather["distance"] = valid_weather.apply(
        lambda row: geodesic((lat, lon), (row["latitude"], row["longitude"])).km
        if pd.notna(row["latitude"]) and pd.notna(row["longitude"]) else np.nan,
        axis=1
    )
    
    # Drop rows where distance couldn't be computed
    valid_weather = valid_weather.dropna(subset=["distance"])
    
    if valid_weather.empty:
        return np.nan

    return valid_weather.loc[valid_weather["distance"].idxmin(), "station_code"]

def merge_all_data(weather, merged_sessions):
    merged_sessions = merged_sessions.dropna(subset=["latitude", "longitude"]).copy()
    merged_sessions["session_date"] = pd.to_datetime(
        merged_sessions["start_datetime"], errors="coerce"
    ).dt.date
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce").dt.date

    print("Matching sessions with nearest weather stations...")
    merged_sessions["nearest_station_code"] = merged_sessions.apply(
        lambda row: find_nearest_station(row["latitude"], row["longitude"], weather), axis=1
    )

    combined = pd.merge(
        merged_sessions,
        weather,
        left_on=["nearest_station_code", "session_date"],
        right_on=["station_code", "date"],
        how="left"
    )
    return combined

def develop_weather_sensitivity_score(df):
    df["weather_sensitivity_score"] = 0

    df.loc[df["precipitation_mm"] > 10, "weather_sensitivity_score"] += 2
    df.loc[df["precipitation_mm"] > 30, "weather_sensitivity_score"] += 3
    df.loc[(df["temp_avg_c"] < 0) | (df["temp_avg_c"] > 35), "weather_sensitivity_score"] += 2
    df.loc[(df["temp_max_c"] > 40) & (df["precipitation_mm"] > 20), "weather_sensitivity_score"] += 2

    max_score = df["weather_sensitivity_score"].max()
    if max_score > 0:
        df["weather_sensitivity_score"] = (df["weather_sensitivity_score"] / max_score) * 10

    return df

