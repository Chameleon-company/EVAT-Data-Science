"""
EV Charging Infrastructure & Congestion Clustering Pipeline
-----------------------------------------------------------
This script:
1. Loads and preprocesses travel, congestion, EV registration, charger, population, and dwelling datasets.
2. Aggregates and merges data (with fuzzy suburb matching).
3. Engineers features for clustering.
4. Runs KMeans clustering and labels cluster profiles.
5. Generates an interactive Folium map of clustered suburbs.
"""

# =========================
# Imports
# =========================
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from thefuzz import process

# =========================
# Helper Functions
# =========================
def clean_suburb(s):
    if pd.isnull(s):
        return ""
    return str(s).lower().strip().replace(".", "").replace("-", " ").replace("  ", " ")

def fuzzy_merge(left_df, right_df, left_on, right_on, threshold=90, limit=1):
    """
    Perform fuzzy merge between two dataframes based on suburb names.
    """
    left_df[left_on] = left_df[left_on].apply(clean_suburb)
    right_df[right_on] = right_df[right_on].apply(clean_suburb)

    matches = []
    right_choices = right_df[right_on].unique()

    for name in left_df[left_on].unique():
        match = process.extractOne(name, right_choices)
        if match and match[1] >= threshold:
            matches.append({left_on: name, "match_name": match[0], "score": match[1]})

    matches_df = pd.DataFrame(matches)
    merged_left = pd.merge(left_df, matches_df, on=left_on, how="left")
    merged = pd.merge(
        merged_left,
        right_df,
        left_on="match_name",
        right_on=right_on,
        how="left",
        suffixes=("", "_right"),
    )

    return merged

# =========================
# Step 1: Load Datasets
# =========================
chargers_df = pd.read_csv("Charger-Info.csv")
ev_reg_df = pd.read_csv("vehicle_registrations.csv")
travel_df = pd.read_csv("ml_ev_charging_dataset.csv")
congestion_df = pd.read_csv("road_congestion.csv")
population_df = pd.read_csv("Suburb_Population.csv")
dwelling_df = pd.read_csv("Info_for_PCZ.csv")
stations_df = pd.read_csv("stations_per_town.csv")
coordinates_df = pd.read_csv("Co-oridnates.csv")

# =========================
# Step 2: Preprocessing
# =========================
# Chargers
chargers_df["Suburb"] = chargers_df["Suburb"].str.title().str.strip()
chargers_per_suburb = chargers_df.groupby("Suburb").size().reset_index(name="Public_Chargers")

# EV registrations
ev_per_postcode = ev_reg_df.groupby("POSTCODE")["TOTAL1"].sum().reset_index(name="EV_Count")
postcode_map = pd.DataFrame({
    "Suburb": chargers_df["Suburb"],
    "POSTCODE": chargers_df["Postal Code"].astype(str),
})
ev_per_postcode["POSTCODE"] = ev_per_postcode["POSTCODE"].astype(str)
ev_per_suburb = pd.merge(ev_per_postcode, postcode_map, on="POSTCODE")
ev_per_suburb = ev_per_suburb.groupby("Suburb")["EV_Count"].sum().reset_index()

# Travel data
travel_df["Suburb"] = travel_df["Address"].str.extract(r",\s*([\w\s]+),\s*VIC")[0].str.title().str.strip()
travel_agg = travel_df.groupby("Suburb")[["Distance_km", "ETA_min"]].mean().reset_index()

# Congestion
congestion_df.replace(-1, np.nan, inplace=True)
congestion_agg = (
    congestion_df.groupby("Location")[["QT_VOLUME_24HOUR"]]
    .mean()
    .reset_index()
    .rename(columns={"Location": "Suburb", "QT_VOLUME_24HOUR": "Avg_Congestion"})
)
congestion_agg["Suburb"] = congestion_agg["Suburb"].str.title().str.strip()

# Population
population_df.rename(columns={"Town": "Suburb"}, inplace=True)
population_df["Suburb"] = population_df["Suburb"].str.title().str.strip()

# Dwellings
dwelling_df.rename(
    columns={
        "Town": "Suburb",
        "All Private Dwellings": "Dwellings",
        "Median Weekly Household Income": "Income",
        "Average Motor Vehicles per Dwelling": "Vehicles_per_Dwelling",
    },
    inplace=True,
)
dwelling_df["Income"] = dwelling_df["Income"].replace("[\$,]", "", regex=True).astype(float)
dwelling_df["Suburb"] = dwelling_df["Suburb"].str.title().str.strip()

# Stations
stations_df.rename(columns={"Town": "Suburb", "Number of Charging Stations": "Station_Count"}, inplace=True)
stations_df["Suburb"] = stations_df["Suburb"].str.title().str.strip()

# =========================
# Step 3: Fuzzy Merge
# =========================
merged_full = fuzzy_merge(population_df, dwelling_df, "Suburb", "Suburb", threshold=90)
merged_full.drop(columns=["match_name", "score"], inplace=True)

datasets_to_merge = [chargers_per_suburb, ev_per_suburb, travel_agg, congestion_agg, stations_df]
for df in datasets_to_merge:
    merged_full = fuzzy_merge(merged_full, df, "Suburb", "Suburb", threshold=90)
    merged_full.drop(columns=["match_name", "score"], inplace=True, errors="ignore")

# Clean up numeric fields
for col in ["Dwellings", "Income", "Vehicles_per_Dwelling"]:
    if col in merged_full.columns:
        merged_full[col] = (
            merged_full[col]
            .astype(str)
            .str.replace(",", "")
            .str.replace("$", "")
            .str.strip()
        )
        merged_full[col] = pd.to_numeric(merged_full[col], errors="coerce")

# Fill missing values
merged_full.fillna({
    "Public_Chargers": 0,
    "EV_Count": 0,
    "Distance_km": merged_full["Distance_km"].median(),
    "ETA_min": merged_full["ETA_min"].median(),
    "Avg_Congestion": merged_full["Avg_Congestion"].median(),
    "Dwellings": merged_full["Dwellings"].median(),
    "Income": merged_full["Income"].median(),
    "Vehicles_per_Dwelling": merged_full["Vehicles_per_Dwelling"].median(),
    "Station_Count": 0,
}, inplace=True)

# Derived features
merged_full["Charger_to_Pop_Ratio"] = merged_full["Public_Chargers"] / merged_full["Population"]
merged_full["EVs_per_Public_Charger"] = merged_full["EV_Count"] / (merged_full["Public_Chargers"] + 1e-5)

# =========================
# Step 4: Clustering
# =========================
features = [
    "Population", "Dwellings", "Income", "Vehicles_per_Dwelling",
    "Public_Chargers", "EV_Count", "Distance_km", "ETA_min",
    "Avg_Congestion", "Charger_to_Pop_Ratio", "EVs_per_Public_Charger"
]

X = merged_full[features].replace([np.inf, -np.inf], np.nan).dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
sse = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(2, 15), sse, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("SSE (Inertia)")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.show()

# Final clustering
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
merged_full["Cluster"] = clusters

# Cluster labels
def label_cluster(row):
    if row["EV_Count"] > 50 and row["Public_Chargers"] <= 3:
        return "Urban EV-Ready, Charger Shortage ⚡"
    elif row["EV_Count"] < 10 and row["Public_Chargers"] < 1:
        return "Low EV Adoption, Growth Potential 🌱"
    elif row["EV_Count"] == 0 and row["Distance_km"] > 8:
        return "Remote Area, Infrastructure Gap ❌"
    else:
        return "Moderate Priority Area"

merged_full["Cluster_Label"] = merged_full.apply(label_cluster, axis=1)

# =========================
# Step 5: Visualization
# =========================
cluster_colors = {0: "blue", 1: "green", 2: "red", 3: "purple", 4: "orange"}

merged_full["Suburb_clean"] = merged_full["Suburb"].str.strip().str.lower()
coordinates_df["suburb_clean"] = coordinates_df["suburb"].str.strip().str.lower()

lat_dict = dict(zip(coordinates_df["suburb_clean"], coordinates_df["latitude"]))
lon_dict = dict(zip(coordinates_df["suburb_clean"], coordinates_df["longitude"]))

m = folium.Map(location=[-37.8, 144.9], zoom_start=10)
for _, row in merged_full.iterrows():
    suburb_key = row["Suburb_clean"]
    lat, lon = lat_dict.get(suburb_key), lon_dict.get(suburb_key)
    if lat and lon:
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=cluster_colors.get(row["Cluster"], "gray"),
            popup=f'{row["Suburb"]} - {row["Cluster_Label"]}',
            fill=True,
        ).add_to(m)

m.save("ev_cluster_map.html")

# =========================
# Step 6: Save Outputs
# =========================
merged_full.to_csv("final_merged_for_clustering.csv", index=False)
merged_full.to_csv("clustered_suburbs.csv", index=False)

print("✅ Pipeline complete. Results saved to clustered_suburbs.csv and map saved as ev_cluster_map.html")

