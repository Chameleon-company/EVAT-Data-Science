from flask import Blueprint, request, jsonify, current_app
from app.services.user_service import UserService

user_controller = Blueprint('user_controller', __name__)

@user_controller.route('/users', methods=['POST'])
def add_user():
    data = request.json
    username = data.get('username')
    email = data.get('email')
    
    try:
        user_service = UserService(current_app.user_model, current_app.station_model)
        user_id = user_service.add_user(username, email)
        return jsonify({"message": f"User '{username}' added", "id": str(user_id)}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@user_controller.route('/users', methods=['GET'])
def get_users():
    user_service = UserService(current_app.user_model, current_app.station_model)
    users = user_service.get_users()
    return jsonify(users), 200

@user_controller.route('/users/<user_id>', methods=['DELETE'])
def remove_user(user_id):
    user_service = UserService(current_app.user_model, current_app.station_model)
    result = user_service.remove_user(user_id)
    if result.deleted_count:
        return jsonify({"message": f"User with ID '{user_id}' removed"}), 200
    else:
        return jsonify({"error": "User not found"}), 404

@user_controller.route('/users/<user_id>/favorites', methods=['POST'])
def add_favorite_station(user_id):
    data = request.json
    station_name = data.get('station_name')

    try:
        user_service = UserService(current_app.user_model, current_app.station_model)
        user_service.add_favorite_station(user_id, station_name)
        return jsonify({"message": f"Station '{station_name}' added to favorites"}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@user_controller.route('/users/<user_id>/favorites', methods=['DELETE'])
def remove_favorite_station(user_id):
    station_name = request.args.get('station_name')

    try:
        user_service = UserService(current_app.user_model, current_app.station_model)
        user_service.remove_favorite_station(user_id, station_name)
        return jsonify({"message": f"Station '{station_name}' removed from favorites"}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@user_controller.route('/users/<user_id>/favorites', methods=['GET'])
def get_favorite_stations(user_id):
    try:
        user_service = UserService(current_app.user_model, current_app.station_model)
        stations = user_service.get_favorite_stations(user_id)
        return jsonify(stations), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
