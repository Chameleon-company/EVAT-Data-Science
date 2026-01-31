import pandas as pd
from shapely import wkt
from shapely.geometry import Point
import geopandas as gpd
import joblib
from flask import Flask, request, jsonify

try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("Warning: flask-cors not installed. CORS support disabled.")
    print("Install with: pip install flask-cors")

model = joblib.load("ev_model.pkl")

FEATURES = [
    "Year",
    "SHAPE_Length",
    "dist_to_nearest_ev_m",
    "ev_within_500m",
    "avg_temp",
    "total_prcp",
]

METRIC_EPSG = 32755

traffic_df = pd.read_csv("Traffic data.csv")
traffic_df["geometry"] = traffic_df["geometry"].apply(wkt.loads)

traffic_gdf = gpd.GeoDataFrame(
    traffic_df,
    geometry="geometry",
    crs="EPSG:4326"
).to_crs(epsg=METRIC_EPSG)

ev_df = pd.read_csv("EV stations data.csv")
ev_df = ev_df.dropna(subset=["Latitude", "Longitude"])

ev_gdf = gpd.GeoDataFrame(
    ev_df,
    geometry=gpd.points_from_xy(ev_df["Longitude"], ev_df["Latitude"]),
    crs="EPSG:4326"
).to_crs(epsg=METRIC_EPSG)

ev_coords = ev_df[["InfrastructureID", "Latitude", "Longitude"]]

weather_df = pd.read_csv("weather_by_station_2023.csv")
weather_df = weather_df.merge(ev_coords, on="InfrastructureID", how="left")

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

def _point_in_metric_crs(lat: float, lon: float, target_crs) -> Point:
    return gpd.GeoSeries(
        [Point(lon, lat)],
        crs="EPSG:4326"
    ).to_crs(target_crs).iloc[0]

def get_shape_length_from_coords(lat: float, lon: float) -> float:
    pt = _point_in_metric_crs(lat, lon, traffic_gdf.crs)
    distances = traffic_gdf.distance(pt)
    nearest_idx = distances.idxmin()
    shape_length = traffic_gdf.loc[nearest_idx, "SHAPE_Length"]
    return float(shape_length)

def get_ev_features_from_coords(lat: float, lon: float) -> tuple[float, int]:
    pt = _point_in_metric_crs(lat, lon, ev_gdf.crs)
    distances = ev_gdf.distance(pt)
    dist_to_nearest_ev_m = float(distances.min())
    ev_within_500m = int((distances <= 500).sum())
    return dist_to_nearest_ev_m, ev_within_500m

def get_weather_features_from_coords(lat: float, lon: float) -> tuple[float, float]:
    pt = _point_in_metric_crs(lat, lon, weather_gdf.crs)
    distances = weather_gdf.distance(pt)
    nearest_idx = distances.idxmin()
    avg_temp = float(weather_gdf.loc[nearest_idx, "avg_temp"])
    total_prcp = float(weather_gdf.loc[nearest_idx, "total_prcp"])
    return avg_temp, total_prcp

app = Flask(__name__)
if CORS_AVAILABLE:
    CORS(app)
else:
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "EV model API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Request body is required"}), 400
        
        year = data.get("year")
        start_lat = data.get("start_lat")
        start_lon = data.get("start_lon")
        
        if year is None or start_lat is None or start_lon is None:
            return jsonify({
                "error": "Missing required fields: year, start_lat, start_lon"
            }), 400
        
        year = int(year)
        start_lat = float(start_lat)
        start_lon = float(start_lon)
        
        shape_length = get_shape_length_from_coords(start_lat, start_lon)
        dist_to_nearest_ev_m, ev_within_500m = get_ev_features_from_coords(
            start_lat, start_lon
        )
        avg_temp, total_prcp = get_weather_features_from_coords(
            start_lat, start_lon
        )

        feature_data = pd.DataFrame([[
            year,
            shape_length,
            dist_to_nearest_ev_m,
            ev_within_500m,
            avg_temp,
            total_prcp,
        ]], columns=FEATURES)

        pred = model.predict(feature_data)[0]

        return jsonify({
            "year": year,
            "start_lat": start_lat,
            "start_lon": start_lon,
            "dist_to_nearest_ev_m": dist_to_nearest_ev_m,
            "ev_within_500m": ev_within_500m,
            "avg_temp": avg_temp,
            "total_prcp": total_prcp,
            "used_SHAPE_Length": shape_length,
            "prediction": float(pred),
        })
    
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    print("=" * 50)
    print("Starting EV Traffic & Weather Model API...")
    print("Server will be available at http://localhost:5001")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5001, debug=True)
