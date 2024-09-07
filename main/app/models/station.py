from pymongo import MongoClient
from bson import ObjectId

class Station:
    def __init__(self, db):
        self.collection = db['charging_stations']

    def to_dict(self, station):
        return {
            "id": str(station["_id"]),
            "cost": station["cost"],
            "charging_points": station["charging_points"],
            "pay_at_location": station["pay_at_location"],
            "membership_required": station["membership_required"],
            "access_key_required": station["access_key_required"],
            "is_operational": station["is_operational"],
            "latitude": station["latitude"],
            "longitude": station["longitude"],
            "operator": station["operator"],
            "connection_type": station["connection_type"],
            "current_type": station["current_type"],
            "charging_points_flag": station["charging_points_flag"]
        }

    def find_all(self, page, size):
        offset = (page - 1) * size
        return list(self.collection.find().skip(offset).limit(size))

    def find_by_id(self, station_id):
        return self.collection.find_one({"_id": ObjectId(station_id)})
