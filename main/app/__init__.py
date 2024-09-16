from flask import Flask
from pymongo import MongoClient
from app.models.station import Station
from app.models.user import User
from app.controllers.station_controller import station_controller
from app.controllers.user_controller import user_controller
from dotenv import load_dotenv
import os

def create_app():
    app = Flask(__name__)

    # Load environment variables from .env file
    load_dotenv()

    # Access environment variables
    app.config['GOOGLE_MAP_API_KEY'] = os.getenv('GOOGLE_MAP_API_KEY')
    app.config['DATABASE_URL'] = os.getenv('DATABASE_URL')

    # Set up MongoDB connection
    client = MongoClient(app.config['DATABASE_URL'], tls=True, tlsAllowInvalidCertificates=True)

    db = client['EVAT']

    # Initialize the models
    app.charging_stations = Station(db)
    app.users = User(db)

    # Register the controller blueprints
    app.register_blueprint(station_controller)
    app.register_blueprint(user_controller)

    return app
