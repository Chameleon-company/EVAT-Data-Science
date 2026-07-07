import openrouteservice
import requests
import yaml

# Load API keys from config
with open('config/api_keys.yaml', 'r') as file:
    api_keys = yaml.safe_load(file)

# Initialize OpenRouteService client
ors_client = openrouteservice.Client(key=api_keys['openrouteservice'])

# Function to get weather data
def get_weather(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_keys['openweathermap']}&units=metric"
    response = requests.get(url)
    return response.json()

# Function to get charging stations
def get_charging_stations(lat, lon):
    url = f"https://api.openchargemap.io/v3/poi/?output=json&latitude={lat}&longitude={lon}&distance=10&maxresults=5&key={api_keys['openchargemap']}"
    response = requests.get(url)
    return response.json()

# Function to get traffic data
def get_traffic_data(origin, destination):
    url = f"https://api.tomtom.com/traffic/services/4/incidentDetails?bbox={origin[1]},{origin[0]},{destination[1]},{destination[0]}&key={api_keys['tomtom']}"
    response = requests.get(url)
    return response.json()

# Example usage
origin = (51.5074, -0.1278)  # London
destination = (51.5074, -0.0878)  # Near London

# Get route directions
route = ors_client.directions(
    coordinates=[origin, destination],
    profile='driving-car',
    format='geojson'
)

# Get weather data along the route
weather_data = [get_weather(lat, lon) for lon, lat in route['features'][0]['geometry']['coordinates']]

# Get charging stations near the destination
charging_stations = get_charging_stations(destination[0], destination[1])

# Get traffic data
traffic = get_traffic_data(origin, destination)

# Output results
print("Weather Data:", weather_data)
print("Charging Stations:", charging_stations)
print("Traffic Data:", traffic)
