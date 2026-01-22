import pandas as pd
from shapely import wkt
from shapely.geometry import Point
import geopandas as gpd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel


# 1. Load model and create feature list

model = joblib.load("ev_model.pkl")

FEATURES = [
    "Year",
    "SHAPE_Length",
    "dist_to_nearest_ev_m",
    "ev_within_500m",
    "avg_temp",
    "total_prcp",
]

# We’ll use a metric CRS for distance in meters (adjust if needed)
METRIC_EPSG = 32755  # example: UTM zone 55S (works for much of VIC)

# 2. Load traffic data 

traffic_df = pd.read_csv("Traffic data.csv")
traffic_df["geometry"] = traffic_df["geometry"].apply(wkt.loads)

traffic_gdf = gpd.GeoDataFrame(
    traffic_df,
    geometry="geometry",
    crs="EPSG:4326"
).to_crs(epsg=METRIC_EPSG)

# 3. Load EV stations

ev_df = pd.read_csv("EV stations data.csv")
ev_df = ev_df.dropna(subset=["Latitude", "Longitude"])

ev_gdf = gpd.GeoDataFrame(
    ev_df,
    geometry=gpd.points_from_xy(ev_df["Longitude"], ev_df["Latitude"]),
    crs="EPSG:4326"
).to_crs(epsg=METRIC_EPSG)

# Create EV table with its latitude and longtitude 
ev_coords = ev_df[["InfrastructureID", "Latitude", "Longitude"]]

# 4. Load weather per station and create GeoDataFrame
#    weather_by_station_2023.csv: date, TMAX, TMIN, TAVG, PRCP, InfrastructureID


weather_df = pd.read_csv("weather_by_station_2023.csv")

# Attach lat/lon via InfrastructureID
weather_df = weather_df.merge(ev_coords, on="InfrastructureID", how="left")

# Aggregate to station-level features
weather_station_features = (
    weather_df
    .groupby(["InfrastructureID", "Latitude", "Longitude"])[["TAVG", "PRCP"]]
    .agg(
        avg_temp=("TAVG", "mean"),
        total_prcp=("PRCP", "sum")
    )
    .reset_index()
)

weather_gdf = gpd.GeoDataFrame(
    weather_station_features,
    geometry=gpd.points_from_xy(
        weather_station_features["Longitude"],
        weather_station_features["Latitude"]
    ),
    crs="EPSG:4326"
).to_crs(epsg=METRIC_EPSG)

# 5. Helper functions

def _point_in_metric_crs(lat: float, lon: float, target_crs) -> Point:
    """Convert (lat, lon) to a Point in the target CRS."""
    return gpd.GeoSeries(
        [Point(lon, lat)],
        crs="EPSG:4326"
    ).to_crs(target_crs).iloc[0]


def get_shape_length_from_coords(lat: float, lon: float) -> float:
    """Find nearest road segment and return its SHAPE_Length."""
    pt = _point_in_metric_crs(lat, lon, traffic_gdf.crs)
    distances = traffic_gdf.distance(pt)
    nearest_idx = distances.idxmin()
    shape_length = traffic_gdf.loc[nearest_idx, "SHAPE_Length"]
    return float(shape_length)


def get_ev_features_from_coords(lat: float, lon: float) -> tuple[float, int]:
    """
    Return:
      dist_to_nearest_ev_m: distance to nearest EV station (meters)
      ev_within_500m: number of EV stations within 500m
    """
    pt = _point_in_metric_crs(lat, lon, ev_gdf.crs)
    distances = ev_gdf.distance(pt)
    dist_to_nearest_ev_m = float(distances.min())
    ev_within_500m = int((distances <= 500).sum())
    return dist_to_nearest_ev_m, ev_within_500m


def get_weather_features_from_coords(lat: float, lon: float) -> tuple[float, float]:
    """
    Return (avg_temp, total_prcp) for the nearest weather station.
    Uses precomputed station-level avg_temp and total_prcp.
    """
    pt = _point_in_metric_crs(lat, lon, weather_gdf.crs)
    distances = weather_gdf.distance(pt)
    nearest_idx = distances.idxmin()
    avg_temp = float(weather_gdf.loc[nearest_idx, "avg_temp"])
    total_prcp = float(weather_gdf.loc[nearest_idx, "total_prcp"])
    return avg_temp, total_prcp


# 6. Create FastAPI


class CoordRequest(BaseModel):
    Year: int
    start_lat: float
    start_lon: float


app = FastAPI(title="EV Traffic & Weather Model API")


@app.get("/")
def root():
    return {"message": "EV model API is running"}


@app.post("/predict_from_coords")
def predict_from_coords(req: CoordRequest):
    # 1) Get shape_length
    shape_length = get_shape_length_from_coords(req.start_lat, req.start_lon)

    # 2) get distance to nearest EV station and number ev within 500m
    dist_to_nearest_ev_m, ev_within_500m = get_ev_features_from_coords(
        req.start_lat, req.start_lon
    )

    # 3) get weather features
    avg_temp, total_prcp = get_weather_features_from_coords(
        req.start_lat, req.start_lon
    )

    # 4) build dataframe
    data = pd.DataFrame([[
        req.Year,
        shape_length,
        dist_to_nearest_ev_m,
        ev_within_500m,
        avg_temp,
        total_prcp,
    ]], columns=FEATURES)

    # 5) run model
    pred = model.predict(data)[0]

    # 6) return all features + prediction
    return {
        "Year": req.Year,
        "start_lat": req.start_lat,
        "start_lon": req.start_lon,
        "dist_to_nearest_ev_m": dist_to_nearest_ev_m,
        "ev_within_500m": ev_within_500m,
        "avg_temp": avg_temp,
        "total_prcp": total_prcp,
        "used_SHAPE_Length": shape_length,
        "prediction": float(pred),
    }
