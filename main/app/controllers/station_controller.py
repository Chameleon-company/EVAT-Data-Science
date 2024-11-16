from flask import Blueprint, request, current_app
from flask_restx import Api, Resource, fields
from app.services.station_service import StationService
import requests
import os
from app.swagger import api

station_controller = Blueprint('station_controller', __name__)

# Create namespace
ns = api.namespace('stations', description='Charging station operations')

station_model = api.model('Station', {
    'id': fields.String(description='Station ID'),
    'cost': fields.Float(description='Charging cost'),
    'charging_points': fields.Integer(description='Number of charging points'),
    'pay_at_location': fields.Boolean(description='Payment available at location'),
    'membership_required': fields.Boolean(description='Membership requirement'),
    'access_key_required': fields.Boolean(description='Access key requirement'),
    'is_operational': fields.Boolean(description='Operational status'),
    'latitude': fields.Float(description='Station latitude'),
    'longitude': fields.Float(description='Station longitude'),
    'operator': fields.String(description='Station operator'),
    'connection_type': fields.String(description='Type of connection'),
    'current_type': fields.String(description='Type of current'),
    'charging_points_flag': fields.Boolean(description='Charging points availability')
})

station_distance_model = api.model('StationDistance', {
    'station_info': fields.Nested(station_model),
    'distance_text': fields.String(description='Distance in text format'),
    'distance_exact_value': fields.Integer(description='Distance in meters'),
    'duration': fields.String(description='Travel duration')
})

@ns.route('')
class StationList(Resource):
    @ns.doc('list_stations',
            params={
                'page': {'description': 'Page number', 'type': 'integer', 'default': 1},
                'size': {'description': 'Items per page', 'type': 'integer', 'default': 10}
            })
    @ns.response(200, 'Success', [station_model])
    def get(self):
        page = int(request.args.get('page', 1))
        size = int(request.args.get('size', 10))

        station_service = StationService(current_app.charging_stations)
        stations = station_service.get_stations(page, size)

        return stations, 200

@ns.route('/nearest_station')
class NearestStation(Resource):
    @ns.doc('get_nearest_station',
            params={
                'threshold': {'description': 'Search radius in kilometers', 'type': 'float', 'default': 0.1}
            })
    @ns.response(200, 'Success', [station_distance_model])
    @ns.response(404, 'No stations found')
    @ns.response(500, 'Server error')
    def get(self):
        GOOGLE_MAP_API_KEY = os.getenv('GOOGLE_MAP_API_KEY')
        # Fetch user location via Geolocation API
        geo_url = f'https://www.googleapis.com/geolocation/v1/geolocate?key={GOOGLE_MAP_API_KEY}'
        geo_response = requests.post(geo_url, json={})

        if geo_response.status_code != 200:
            api.abort(500, 'Unable to fetch user location')

        location_data = geo_response.json()
        user_lat = location_data['location']['lat']
        user_lng = location_data['location']['lng']

        # Get the threshold from query parameters (default to 0.1 if not provided)
        threshold = request.args.get('threshold', default=0.1, type=float)

        # Find nearest charging stations using MongoDB geospatial query with the threshold
        station_service = StationService(current_app.charging_stations)
        nearest_stations = station_service.get_nearest_station(user_lat, user_lng, threshold)

        if not nearest_stations:
            api.abort(404, 'No charging stations found nearby')

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
                api.abort(500, 'Unable to calculate distance for station')

            # Parse distance data
            distance_data = distance_response.json()
            distance_info = distance_data['rows'][0]['elements'][0]

            # Append station info with distance details
            station_distances.append({
                'station_info': station,
                'distance_text': distance_info['distance']['text'],
                'distance_exact_value': distance_info['distance']['value'],
                'duration': distance_info['duration']['text']
            })

        # Return the list of stations with distance info
        return station_distances