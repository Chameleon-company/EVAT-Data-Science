from flask import Flask
from pymongo import MongoClient
from app.models.station import Station
from app.models.user import User
from app.controllers.station_controller import station_controller
from app.controllers.user_controller import user_controller

def create_app():
    app = Flask(__name__)

    # Set up MongoDB connection
    client = MongoClient('mongodb+srv://EVAT:EVAT123@cluster0.5axoq.mongodb.net/', tls=True, tlsAllowInvalidCertificates=True)

    db = client['EVAT']

    # Initialize the models
    app.charging_stations = Station(db)
    app.users = User(db)

    # Register the controller blueprints
    app.register_blueprint(station_controller)
    app.register_blueprint(user_controller)

    return app
