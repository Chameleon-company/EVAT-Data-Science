# app.py
from flask import Flask, jsonify, request
from pymongo import MongoClient
from datetime import datetime
import pandas as pd
from bson import ObjectId
import os

app = Flask(__name__)

# MongoDB connection
def get_database():
    USERNAME = "adeshpatil0907"
    PASSWORD = "pXuwLJEt9YcdEQef"
    CONNECTION_STRING = f"mongodb+srv://{USERNAME}:{PASSWORD}@ev.l2r4t.mongodb.net/"
    
    client = MongoClient(CONNECTION_STRING)
    return client['EV_charging_station']

# Updated data model with new fields
extended_fields = {
    "location_data": {
        "population_density": float,
        "distance_to_highway": float,
        "daily_traffic_volume": int,
        "nearby_amenities": {
            "restaurants": int,
            "shopping_centers": int,
            "parking_spots": int
        },
        "land_use_type": str,
        "median_income": float,
        "climate_zone": str
    },
    "usage_data": {
        "peak_hours": str,
        "avg_charging_duration": float,
        "daily_utilization_rate": float,
        "weekly_peak_day": str,
        "avg_wait_time": float,
        "installation_date": datetime,
        "last_maintenance_date": datetime,
        "downtime_hours_per_month": float
    },
    "vehicle_data": {
        "most_common_ev_model": str,
        "avg_battery_capacity": float,
        "max_charging_speed": float,
        "compatible_brands": list
    }
}

# Routes
@app.route('/api/stations', methods=['GET'])
def get_stations():
    """Get all charging stations"""
    try:
        db = get_database()
        stations = list(db.charging_stations.find({}, {'_id': str}))
        return jsonify({
            'success': True,
            'count': len(stations),
            'stations': stations
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stations/<string:station_id>', methods=['GET'])
def get_station(station_id):
    """Get a specific charging station"""
    try:
        db = get_database()
        station = db.charging_stations.find_one({'_id': ObjectId(station_id)})
        if station:
            station['_id'] = str(station['_id'])
            return jsonify({'success': True, 'station': station})
        return jsonify({'success': False, 'error': 'Station not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stations/<string:station_id>/update', methods=['PUT'])
def update_station(station_id):
    """Update a charging station with enriched data"""
    try:
        db = get_database()
        data = request.json
        
        # Validate required fields
        update_data = {
            'location_data': {
                'population_density': data.get('population_density'),
                'distance_to_highway': data.get('distance_to_highway'),
                'daily_traffic_volume': data.get('daily_traffic_volume'),
                'nearby_amenities': {
                    'restaurants': data.get('restaurants', 0),
                    'shopping_centers': data.get('shopping_centers', 0),
                    'parking_spots': data.get('parking_spots', 0)
                },
                'land_use_type': data.get('land_use_type'),
                'median_income': data.get('median_income'),
                'climate_zone': data.get('climate_zone', 'Temperate Oceanic (Cfb)')
            },
            'usage_data': {
                'peak_hours': data.get('peak_hours'),
                'avg_charging_duration': data.get('avg_charging_duration'),
                'daily_utilization_rate': data.get('daily_utilization_rate'),
                'weekly_peak_day': data.get('weekly_peak_day'),
                'avg_wait_time': data.get('avg_wait_time'),
                'installation_date': datetime.strptime(data.get('installation_date', '2023-01-01'), '%Y-%m-%d'),
                'last_maintenance_date': datetime.strptime(data.get('last_maintenance_date', '2023-01-01'), '%Y-%m-%d'),
                'downtime_hours_per_month': data.get('downtime_hours_per_month', 0)
            },
            'vehicle_data': {
                'most_common_ev_model': data.get('most_common_ev_model'),
                'avg_battery_capacity': data.get('avg_battery_capacity'),
                'max_charging_speed': data.get('max_charging_speed'),
                'compatible_brands': data.get('compatible_brands', [])
            }
        }
        
        result = db.charging_stations.update_one(
            {'_id': ObjectId(station_id)},
            {'$set': update_data}
        )
        
        if result.modified_count > 0:
            return jsonify({
                'success': True,
                'message': 'Station updated successfully'
            })
        return jsonify({'success': False, 'error': 'Station not found'}), 404
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stations/bulk-update', methods=['POST'])
def bulk_update_stations():
    """Bulk update stations with enriched data"""
    try:
        db = get_database()
        data = request.json
        
        if not isinstance(data, list):
            return jsonify({'success': False, 'error': 'Expected array of stations'}), 400
            
        modified_count = 0
        for station in data:
            station_id = station.get('_id')
            if not station_id:
                continue
                
            update_data = {
                'location_data': station.get('location_data', {}),
                'usage_data': station.get('usage_data', {}),
                'vehicle_data': station.get('vehicle_data', {})
            }
            
            result = db.charging_stations.update_one(
                {'_id': ObjectId(station_id)},
                {'$set': update_data}
            )
            modified_count += result.modified_count
            
        return jsonify({
            'success': True,
            'modified_count': modified_count
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stations/update-schema', methods=['POST'])
def update_database_schema():
    """Update database schema with new fields"""
    try:
        db = get_database()
        
        # Add default values for new fields to all documents
        default_update = {
            'location_data': {
                'population_density': 0.0,
                'distance_to_highway': 0.0,
                'daily_traffic_volume': 0,
                'nearby_amenities': {
                    'restaurants': 0,
                    'shopping_centers': 0,
                    'parking_spots': 0
                },
                'land_use_type': 'Unknown',
                'median_income': 0.0,
                'climate_zone': 'Temperate Oceanic (Cfb)'
            },
            'usage_data': {
                'peak_hours': '08:00-10:00, 16:00-18:00',
                'avg_charging_duration': 0.0,
                'daily_utilization_rate': 0.0,
                'weekly_peak_day': 'Monday',
                'avg_wait_time': 0.0,
                'installation_date': datetime.now(),
                'last_maintenance_date': datetime.now(),
                'downtime_hours_per_month': 0.0
            },
            'vehicle_data': {
                'most_common_ev_model': 'Unknown',
                'avg_battery_capacity': 0.0,
                'max_charging_speed': 0.0,
                'compatible_brands': []
            }
        }
        
        result = db.charging_stations.update_many(
            {},
            {'$set': default_update}
        )
        
        return jsonify({
            'success': True,
            'modified_count': result.modified_count,
            'message': 'Database schema updated successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)