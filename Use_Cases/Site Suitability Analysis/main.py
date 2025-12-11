from fastapi import FastAPI, HTTPException
import pandas as pd
from geopy.distance import geodesic

# Load the dataset
df = pd.read_csv("priority_sites.csv")

app = FastAPI(title="EVAT Site Suitability API")

# Helper function to find nearest SA2
def find_nearest_sa2(lat, lng):
    df["distance"] = df.apply(
        lambda row: geodesic((lat, lng), (row["lat"], row["lng"])).km,
        axis=1
    )
    return df.loc[df["distance"].idxmin()]

@app.get("/")
def root():
    return {"message": "Site Suitability API is running!"}

@app.get("/suitability/area/{sa2_name}")
def get_suitability_by_area(sa2_name: str):
    result = df[df["sa2"].str.lower() == sa2_name.lower()]
    if result.empty:
        raise HTTPException(status_code=404, detail="SA2 area not found.")

    row = result.iloc[0]
    return {
        "sa2": row["sa2"],
        "cluster_label": row["cluster_label"],
        "population": row["population"],
        "chargers": row["charger_count"],
        "avg_daily_traffic": row["traffic_volume"],
        "amenity_score": row["amenity_score"],
        "irsad_score": row["irsad_score"],
        "suitability_score": row["suitability_score"],
        "rank": int(row["rank"])
    }

@app.get("/suitability/top")
def get_top_sites(n: int = 10):
    top_sites = df.sort_values("suitability_score", ascending=False).head(n)
    return top_sites[["sa2", "rank", "suitability_score"]].to_dict(orient="records")

@app.get("/suitability/coords")
def get_suitability_by_coords(lat: float, lng: float):
    row = find_nearest_sa2(lat, lng)
    return {
        "sa2": row["sa2"],
        "distance_km": float(row["distance"]),
        "suitability_score": row["suitability_score"],
        "cluster_label": row["cluster_label"],
        "rank": int(row["rank"])
    }
