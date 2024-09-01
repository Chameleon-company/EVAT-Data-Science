from flask import Flask, request, jsonify
import googlemaps

app = Flask(__name__)

def create_gmaps_client(api_key):
    return googlemaps.Client(key=api_key)

def calculate_distance_between_origin_to_destination_in_km(api_key, origin, destination):
    gmaps = create_gmaps_client(api_key)
    # Fetch data from Google Maps
    my_dist = gmaps.distance_matrix(origin, destination)['rows'][0]['elements'][0]
    # Value in km
    outcome = my_dist["distance"]["value"] / 1000
    return outcome

@app.route('/api/calculate_battery', methods=['POST'])
def calculate_battery_consumed():
    # Extract API key from headers
    api_key = request.headers.get('X-API-Key')
    
    if not api_key:
        return jsonify({"error": "API key is required in headers"}), 400

    # Extract data from the request
    data = request.json
    origin = data.get('origin')
    destination = data.get('destination')
    max_range = data.get('max_range')
    battery_size = data.get('battery_size')
    
    if not all([origin, destination, max_range, battery_size]):
        return jsonify({"error": "Missing parameters"}), 400

    # Calculate distance
    distance = calculate_distance_between_origin_to_destination_in_km(api_key, origin, destination)

    # Calculate energy consumption rate (kWh/km)
    energy_consumption_rate = battery_size / max_range

    # Calculate possible needed energy (kWh)
    consumed_energy = energy_consumption_rate * distance

    # Calculate % energy consumed
    percent_consumed_energy = consumed_energy / battery_size * 100

    return jsonify({
        "distance_km": round(distance, 2),
        "energy_consumption_rate_kWh_per_km": round(energy_consumption_rate, 2),
        "consumed_energy_kWh": round(consumed_energy, 2),
        "percent_energy_consumed": round(percent_consumed_energy, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
