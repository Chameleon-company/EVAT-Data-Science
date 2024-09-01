from flask import Blueprint, request, jsonify, current_app
from app.services.station_service import StationService

station_controller = Blueprint('station_controller', __name__)

@station_controller.route('/stations', methods=['POST'])
def add_station():
    data = request.json
    station_name = data.get('station_name')
    address = data.get('address')
    
    try:
        station_service = StationService(current_app.station_model)
        station_id = station_service.add_station(station_name, address)
        return jsonify({"message": f"Station '{station_name}' added", "id": str(station_id)}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@station_controller.route('/stations', methods=['GET'])
def get_stations():
    station_service = StationService(current_app.station_model)
    stations = station_service.get_stations()
    return jsonify(stations), 200

@station_controller.route('/stations/<station_id>', methods=['DELETE'])
def remove_station(station_id):
    station_service = StationService(current_app.station_model)
    result = station_service.remove_station(station_id)
    if result.deleted_count:
        return jsonify({"message": f"Station with ID '{station_id}' removed"}), 200
    else:
        return jsonify({"error": "Station not found"}), 404