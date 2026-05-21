import requests
import googlemaps
from datetime import datetime
from config import GOOGLE_MAPS_API_KEY

gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

def get_route(origin: str, destination: str):
    directions = gmaps.directions(
        origin,
        destination,
        mode="driving",
        units="metric",
        departure_time=datetime.now(),
        traffic_model="best_guess"
    )
    if not directions:
        return None
    leg = directions[0]["legs"][0]
    polyline = directions[0]["overview_polyline"]["points"]
    start = (leg["start_location"]["lat"], leg["start_location"]["lng"])
    end   = (leg["end_location"]["lat"], leg["end_location"]["lng"])

    # fixed: sample elevation at every step coordinate instead of just start and end
    # this gives accurate elevation deltas per segment, especially on hilly routes
    path_coords = [
        (step["start_location"]["lat"], step["start_location"]["lng"])
        for step in leg["steps"]
    ]
    path_coords.append((leg["end_location"]["lat"], leg["end_location"]["lng"]))  # added: include final destination
    elevations = gmaps.elevation_along_path(path_coords, len(path_coords))  # fixed: was [start, end]

    return leg, elevations, polyline

def get_weather(lat: float, lon: float):
    try:  # added: error handling so a weather API failure doesn't crash the whole request
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,wind_speed_10m,wind_direction_10m"
            f"&wind_speed_unit=ms"
        )
        r = requests.get(url, timeout=5).json()  # added: timeout to avoid hanging requests
        current = r["current"]
        return {
            "temp_c": current["temperature_2m"],
            "wind_speed_ms": current["wind_speed_10m"],
            "wind_deg": current["wind_direction_10m"],
        }
    except Exception:
        # added: fallback to neutral weather so trip prediction still works if API is down
        return { "temp_c": 20, "wind_speed_ms": 0, "wind_deg": 0 }

def get_charging_stations(lat: float, lon: float, radius_m: int = 2000):
    try:  # added: error handling so a Places API failure returns empty list instead of crashing
        places = gmaps.places_nearby(
            location=(lat, lon),
            radius=radius_m,
            type="electric_vehicle_charging_station",
            keyword="electric vehicle charging"
        )
        results = []
        for place in places.get("results", [])[:5]:
            results.append({
                "name": place["name"],
                "lat": place["geometry"]["location"]["lat"],
                "lng": place["geometry"]["location"]["lng"],
                "address": place.get("vicinity", ""),
                "rating": place.get("rating", None),
                "open_now": place.get("opening_hours", {}).get("open_now", None),
                "place_id": place["place_id"],
            })
        return results
    except Exception:
        return []  # added: return empty list so response still works if Places API is down