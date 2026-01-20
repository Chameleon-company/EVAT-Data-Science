import pandas as pd
from shapely import wkt
from shapely.geometry import Point
import geopandas as gpd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("ev_model.pkl")

FEATURES = [
    "Year",
    "SHAPE_Length",
    "dist_to_nearest_ev_m",
    "ev_within_500m",
    "avg_temp",
    "total_prcp",
]

traffic_df = pd.read_csv("Traffic data.csv")


traffic_df["geometry"] = traffic_df["geometry"].apply(wkt.loads)

traffic_gdf = gpd.GeoDataFrame(traffic_df, geometry="geometry", crs="EPSG:4326")

def get_shape_length_from_coords(lat: float, lon: float) -> float:
    """
    Given a lat/lon point, find the nearest road segment in traffic_gdf
    and return its SHAPE_Length.
    """
    pt = Point(lon, lat)  

    distances = traffic_gdf.distance(pt)
    nearest_idx = distances.idxmin()

    shape_length = traffic_gdf.loc[nearest_idx, "SHAPE_Length"]
    return float(shape_length)

class CoordRequest(BaseModel):
    Year: int
    start_lat: float
    start_lon: float
    dist_to_nearest_ev_m: float
    ev_within_500m: int
    avg_temp: float
    total_prcp: float
    
app = FastAPI(title="EV Traffic & Weather Model API")

@app.get("/")
def root():
    return {"message": "EV model API is running"}

@app.post("/predict_from_coords")
def predict_from_coords(req: CoordRequest):
    shape_length = get_shape_length_from_coords(req.start_lat, req.start_lon)

    data = pd.DataFrame([[
        req.Year,
        shape_length,
        req.dist_to_nearest_ev_m,
        req.ev_within_500m,
        req.avg_temp,
        req.total_prcp,
    ]], columns=FEATURES)

    pred = model.predict(data)[0]

    return {
        "prediction": float(pred),
        "used_SHAPE_Length": shape_length
    }
