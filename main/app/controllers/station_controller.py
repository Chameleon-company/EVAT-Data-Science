from flask import Blueprint, request, jsonify, current_app
from app.services.station_service import StationService
import requests
import os

station_controller = Blueprint('station_controller', __name__)

@station_controller.route('/stations', methods=['GET'])
def get_stations():
    page = int(request.args.get('page', 1))
    size = int(request.args.get('size', 10))

    station_service = StationService(current_app.charging_stations)
    stations = station_service.get_stations(page, size)
    
    return jsonify(stations), 200

@station_controller.route('/nearest_station', methods=['GET'])
def nearest_station():
    GOOGLE_MAP_API_KEY = os.getenv('GOOGLE_MAP_API_KEY')
    # Fetch user location via Geolocation API
    geo_url = f'https://www.googleapis.com/geolocation/v1/geolocate?key={GOOGLE_MAP_API_KEY}'
    geo_response = requests.post(geo_url, json={})

    if geo_response.status_code != 200:
        return jsonify({'error': 'Unable to fetch user location'}), 500
    
    location_data = geo_response.json()
    user_lat = location_data['location']['lat']
    user_lng = location_data['location']['lng']

    # Get the threshold from query parameters (default to 0.1 if not provided)
    threshold = request.args.get('threshold', default=0.1, type=float)

    # Find nearest charging stations using MongoDB geospatial query with the threshold
    station_service = StationService(current_app.charging_stations)
    nearest_stations = station_service.get_nearest_station(user_lat, user_lng, threshold)

    if not nearest_stations:
        return jsonify({'error': 'No charging stations found nearby'}), 404

    # Calculate distance using Google Distance Matrix API
    distance_matrix_url = 'https://maps.googleapis.com/maps/api/distancematrix/json'

    # Iterate through nearest stations and calculate distance for each
    station_distances = []
    for station in nearest_stations:
        station_lat = station['latitude']
        station_lng = station['longitude']
        
        # Prepare API parameters
        params = {
            'origins': f"{user_lat},{user_lng}",
            'destinations': f"{station_lat},{station_lng}",
            'key': GOOGLE_MAP_API_KEY
        }

        # Send request to Google Distance Matrix API
        distance_response = requests.get(distance_matrix_url, params=params)

        if distance_response.status_code != 200:
            return jsonify({'error': 'Unable to calculate distance for station'}), 500

        # Parse distance data
        distance_data = distance_response.json()
        distance_info = distance_data['rows'][0]['elements'][0]

        # Append station info with distance details
        station_distances.append({
            'station_info': station,
            'distance_text': distance_info['distance']['text'],
            'distance_exact_value': distance_info['distance']['value'],  # Distance in meters
            'duration': distance_info['duration']['text']
        })

    # Return the list of stations with distance info
    return jsonify(station_distances)