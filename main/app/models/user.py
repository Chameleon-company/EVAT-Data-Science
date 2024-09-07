from pymongo import MongoClient
from bson import ObjectId

class User:
    def __init__(self, db):
        self.collection = db['users']

    def to_dict(self, user):
        return {
            "id": str(user["_id"]),
            "username": user["username"],
            "make": user.get("make", ""), 
            "model": user.get("model", ""),  
            "variant": user.get("variant", ""),  
            "connection_type": user.get("connection_type", ""),
            "favorite_stations": user["favorite_stations"]
        }

    def find_all(self):
        return list(self.collection.find())

    def find_by_id(self, user_id):
        return self.collection.find_one({"_id": ObjectId(user_id)})

    def insert(self, username, password):
        user = {
            "username": username,
            "password": password,
            "favorite_stations": []
        }
        result = self.collection.insert_one(user)
        return result.inserted_id

    def add_favorite_station(self, user_id, station_id):
        return self.collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$push": {"favorite_stations": station_id}}
        )

    def remove_favorite_station(self, user_id, station_id):
        return self.collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$pull": {"favorite_stations": station_id}}
        )
