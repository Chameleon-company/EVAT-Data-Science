from flask import Flask
from pymongo import MongoClient
from app.models.station import Station
from app.models.user import User
from app.controllers.station_controller import station_controller
from app.controllers.user_controller import user_controller

def create_app():
    app = Flask(__name__)

    # Set up MongoDB connection
    client = MongoClient('mongodb://localhost:27017/')
    db = client['evat']

    # Initialize the models
    app.station_model = Station(db)
    app.user_model = User(db)

    # Register the controller blueprints
    app.register_blueprint(station_controller)
    app.register_blueprint(user_controller)

    return app
