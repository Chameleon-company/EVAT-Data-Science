from pymongo import MongoClient, GEOSPHERE
from bson import ObjectId

class Station:
    def __init__(self, db):
        self.collection = db['charging_stations']
        self.collection.create_index([("location", GEOSPHERE)])

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

    def find_nearest(self, user_lat, user_lng, threshold):
        nearest_stations = self.collection.find({
            "$expr": {
                "$lt": [
                    {
                        "$sqrt": {
                            "$add": [
                                { "$pow": [{ "$subtract": ["$longitude", user_lng] }, 2] },
                                { "$pow": [{ "$subtract": ["$latitude", user_lat] }, 2] }
                            ]
                        }
                    },
                    threshold
                ]
            }
        })
        
        nearest_stations = [self.to_dict(station) for station in nearest_stations]
        return nearest_stations if nearest_stations else None
