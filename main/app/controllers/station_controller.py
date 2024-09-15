from flask import Blueprint, request, jsonify, current_app
from app.services.station_service import StationService
import requests

station_controller = Blueprint('station_controller', __name__)

API_KEY = 'AIzaSyDbfaYzyw__K8paspxDS6c7Pw5VP6q_R48'

@station_controller.route('/stations', methods=['GET'])
def get_stations():
    page = int(request.args.get('page', 1))
    size = int(request.args.get('size', 10))

    station_service = StationService(current_app.charging_stations)
    stations = station_service.get_stations(page, size)
    
    return jsonify(stations), 200

@station_controller.route('/nearest_station', methods=['GET'])
def nearest_station():
    # Fetch user location via Geolocation API
    geo_url = f'https://www.googleapis.com/geolocation/v1/geolocate?key={API_KEY}'
    geo_response = requests.post(geo_url, json={})

    if geo_response.status_code != 200:
        return jsonify({'error': 'Unable to fetch user location'}), 500
    
    location_data = geo_response.json()
    user_lat = location_data['location']['lat']
    user_lng = location_data['location']['lng']

    print(user_lat, user_lng)
    
    # Find nearest charging station using MongoDB geospatial query
    station_service = StationService(current_app.charging_stations)
    nearest_station = station_service.get_nearest_station(user_lat, user_lng)

    if not nearest_station:
        return jsonify({'error': 'No charging stations found nearby'}), 404

    station_lat = nearest_station['latitude']
    station_lng = nearest_station['longitude']

    # Calculate distance using Google Distance Matrix API
    distance_matrix_url = 'https://maps.googleapis.com/maps/api/distancematrix/json'
    params = {
        'origins': f"{user_lat},{user_lng}",
        'destinations': f"{station_lat},{station_lng}",
        'key': API_KEY
    }

    distance_response = requests.get(distance_matrix_url, params=params)

    if distance_response.status_code != 200:
        return jsonify({'error': 'Unable to calculate distance'}), 500

    distance_data = distance_response.json()
    distance_info = distance_data['rows'][0]['elements'][0]

    # Return station info and distance
    return jsonify({
        'station_info': nearest_station,
        'distance_text': distance_info['distance']['text'],
        'distance_value': distance_info['distance']['value'],
        'duration_text': distance_info['duration']['text']
    })