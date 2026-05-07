from fastapi import FastAPI, HTTPException
import pandas as pd
from geopy.distance import geodesic

# Load the dataset
df = pd.read_csv("priority_sites_version2.csv")

app = FastAPI(title="EVAT Site Suitability API")

# Helper function to find nearest SA2
def find_nearest_sa2(lat, lng):
    df["distance_km"] = df.apply(
        lambda row: geodesic((lat, lng), (row["lat"], row["lon"])).km,
        axis=1
    )
    return df.loc[df["distance_km"].idxmin()]

@app.get("/")
def root():
    return {"message": "Site Suitability API is running!"}

@app.get("/suitability/area/{sa2_name}")
def get_suitability_by_area(sa2_name: str):
    result = df[df["SA2_NAME21"].str.lower() == sa2_name.lower()]
    if result.empty:
        raise HTTPException(status_code=404, detail="SA2 area not found.")

    row = result.iloc[0]
    return {
        "sa2": row["SA2_NAME21"],
        "cluster_label": row["Cluster_Label"],
        "population": row["Total Population"],
        "chargers": row["ChargerCount"],
        "avg_daily_traffic": row["Traffic_200m"],
        "amenity_score": row["AmenityScore_300m"],
        "irsad_score": row["IRSAD Score"],
        "suitability_score": row["SiteScore"],
        "rank": int(row["rank_in_sa2"])
    }

# now returns lat and lon to match the frontend requirement
@app.get("/suitability/top")
def get_top_sites(n: int = 10):
    top_sites = df.sort_values("SiteScore", ascending=False).head(n)
    return top_sites[["SA2_NAME21", "rank_in_sa2", "SiteScore", "lat", "lon"]].rename(columns={
        "SA2_NAME21": "sa2",
        "rank_in_sa2": "rank",
        "SiteScore": "suitability_score"
    }).to_dict(orient="records")

@app.get("/suitability/coords")
def get_suitability_by_coords(lat: float, lng: float):
    row = find_nearest_sa2(lat, lng)
    return {
        "sa2": row["SA2_NAME21"],
        "distance_km": float(row["distance_km"]),
        "suitability_score": row["SiteScore"],
        "cluster_label": row["Cluster_Label"],
        "rank": int(row["rank_in_sa2"])
    }
