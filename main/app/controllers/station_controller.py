from flask import Blueprint, request, jsonify, current_app
from app.services.station_service import StationService

station_controller = Blueprint('station_controller', __name__)

@station_controller.route('/stations', methods=['GET'])
def get_stations():
    page = int(request.args.get('page', 1))
    size = int(request.args.get('size', 10))

    station_service = StationService(current_app.charging_stations)
    stations = station_service.get_stations(page, size)
    
    return jsonify(stations), 200