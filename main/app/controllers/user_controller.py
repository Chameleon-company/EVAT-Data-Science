from flask import Blueprint, request, current_app
from flask_restx import Api, Resource, fields
from app.services.user_service import UserService
from app.swagger import api

user_controller = Blueprint('user_controller', __name__)

# Create namespace
ns = api.namespace('users', description='User operations')

create_user_model = api.model('Create user', {
    'username': fields.String(required=True, description='Username'),
    'password': fields.String(required=True, description='Password'),
})

user_model = api.model('User', {
    'username': fields.String(required=True, description='Username'),
    'password': fields.String(required=True, description='Password'),
    'make': fields.String(required=False, description='Make of the vehicle'),
    'model': fields.String(required=False, description='Model of the vehicle'),
    'variant': fields.String(required=False, description='Variant of the vehicle'),
    'connection_type': fields.String(required=False, description='Connection type'),
    'favorite_stations': fields.List(fields.String, required=False, description='List of favorite stations')
})

favorite_station_model = api.model('FavoriteStation', {
    'station_id': fields.String(required=True, description='Station id')
})

@ns.route('')
class UserResource(Resource):
    @ns.doc('create_user')
    @ns.expect(create_user_model)
    @ns.response(201, 'User successfully created')
    @ns.response(400, 'Validation Error')
    def post(self):
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        try:
            user_service = UserService(current_app.users, current_app.charging_stations)
            user_id = user_service.add_user(username, password)
            return {"message": f"User '{username}' added", "id": str(user_id)}, 201
        except ValueError as e:
            api.abort(400, str(e))
    
    @ns.doc('list_users')
    @ns.response(200, 'Success', [user_model])
    def get(self):
        user_service = UserService(current_app.users, current_app.charging_stations)
        users = user_service.get_users()
        return users, 200

@ns.route('/<string:user_id>/favorites')
class UserFavorites(Resource):
    @ns.doc('add_favorite_station')
    @ns.expect(favorite_station_model)
    @ns.response(201, 'Station successfully added to favorites')
    def post(self, user_id):
        data = request.json
        station_id = data.get('station_id')

        try:
            user_service = UserService(current_app.users, current_app.charging_stations)
            user_service.add_favorite_station(user_id, station_id)
            return {"message": f"Station '{station_id}' added to favorites"}, 201
        except ValueError as e:
            api.abort(400, str(e))

    @ns.doc('remove_favorite_station')
    @ns.expect(favorite_station_model)
    def delete(self, user_id):
        data = request.json
        station_id = data.get('station_id')

        try:
            user_service = UserService(current_app.users, current_app.charging_stations)
            user_service.remove_favorite_station(user_id, station_id)
            return {"message": f"Station '{station_id}' removed from favorites"}, 200
        except ValueError as e:
            api.abort(400, str(e))


    @ns.doc('get_favorite_stations')
    def get(self, user_id):
        try:
            user_service = UserService(current_app.users, current_app.charging_stations)
            stations = user_service.get_favorite_stations(user_id)
            return stations, 200
        except ValueError as e:
            api.abort(400, str(e))