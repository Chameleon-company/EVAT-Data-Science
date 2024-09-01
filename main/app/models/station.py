from pymongo import MongoClient
from bson import ObjectId

class Station:
    def __init__(self, db):
        self.collection = db['stations']

    def to_dict(self, station):
        return {
            "id": str(station["_id"]),
            "station_name": station["station_name"],
            "address": station["address"]
        }

    def find_all(self):
        return list(self.collection.find())

    def find_by_id(self, station_id):
        return self.collection.find_one({"_id": ObjectId(station_id)})
    
    def find_by_name(self, station_name):
        return self.collection.find_one({"station_name": station_name})

    def insert(self, station_name, address):
        station = {
            "station_name": station_name,
            "address": address
        }
        result = self.collection.insert_one(station)
        return result.inserted_id

    def delete(self, station_id):
        return self.collection.delete_one({"_id": ObjectId(station_id)})
