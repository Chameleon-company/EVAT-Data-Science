from flask import Blueprint, request, jsonify, current_app
from app.services.user_service import UserService

user_controller = Blueprint('user_controller', __name__)

@user_controller.route('/users', methods=['POST'])
def add_user():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    try:
        user_service = UserService(current_app.users, current_app.charging_stations)
        user_id = user_service.add_user(username, password)
        return jsonify({"message": f"User '{username}' added", "id": str(user_id)}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@user_controller.route('/users', methods=['GET'])
def get_users():
    user_service = UserService(current_app.users, current_app.charging_stations)
    users = user_service.get_users()
    return jsonify(users), 200

@user_controller.route('/users/<user_id>/favorites', methods=['POST'])
def add_favorite_station(user_id):
    data = request.json
    station_id = data.get('station_id')

    try:
        user_service = UserService(current_app.users, current_app.charging_stations)
        user_service.add_favorite_station(user_id, station_id)
        return jsonify({"message": f"Station '{station_id}' added to favorites"}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@user_controller.route('/users/<user_id>/favorites', methods=['DELETE'])
def remove_favorite_station(user_id):
    data = request.json
    station_id = data.get('station_id')

    try:
        user_service = UserService(current_app.users, current_app.charging_stations)
        user_service.remove_favorite_station(user_id, station_id)
        return jsonify({"message": f"Station '{station_id}' removed from favorites"}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@user_controller.route('/users/<user_id>/favorites', methods=['GET'])
def get_favorite_stations(user_id):
    try:
        user_service = UserService(current_app.users, current_app.charging_stations)
        stations = user_service.get_favorite_stations(user_id)
        return jsonify(stations), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
