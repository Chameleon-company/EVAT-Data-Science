import pymongo
from pymongo import MongoClient
from datetime import datetime
from pprint import pprint

class MongoDBCRUDTester:
    def __init__(self):
        """Initialize MongoDB connection"""
        USERNAME = "adeshpatil0907"
        PASSWORD = "pXuwLJEt9YcdEQef"
        CONNECTION_STRING = f"mongodb+srv://{USERNAME}:{PASSWORD}@ev.l2r4t.mongodb.net/"
        
        try:
            self.client = MongoClient(CONNECTION_STRING)
            self.db = self.client['EV_charging_station']
            self.collection = self.db['charging_stations']
            print("MongoDB connection initialized")
        except Exception as e:
            print(f"Error initializing MongoDB connection: {e}")
            raise

    def test_create_operations(self):
        """Test Create operations"""
        print("\n=== Testing Create Operations ===")
        
        # Insert one document
        single_station = {
            "station_id": "TEST001",
            "name": "Test Station 1",
            "location": {
                "latitude": -37.8136,
                "longitude": 144.9631
            },
            "operator": "Test Operator",
            "charging_points": 4,
            "status": "operational",
            "created_at": datetime.now()
        }
        
        try:
            result = self.collection.insert_one(single_station)
            print(f"Inserted single document with ID: {result.inserted_id}")
        except Exception as e:
            print(f"Error inserting single document: {e}")

        # Insert multiple documents
        multiple_stations = [
            {
                "station_id": "TEST002",
                "name": "Test Station 2",
                "location": {
                    "latitude": -37.8137,
                    "longitude": 144.9632
                },
                "operator": "Test Operator",
                "charging_points": 2,
                "status": "operational",
                "created_at": datetime.now()
            },
            {
                "station_id": "TEST003",
                "name": "Test Station 3",
                "location": {
                    "latitude": -37.8138,
                    "longitude": 144.9633
                },
                "operator": "Test Operator",
                "charging_points": 6,
                "status": "operational",
                "created_at": datetime.now()
            }
        ]
        
        try:
            result = self.collection.insert_many(multiple_stations)
            print(f"Inserted {len(result.inserted_ids)} documents")
        except Exception as e:
            print(f"Error inserting multiple documents: {e}")

    def test_read_operations(self):
        """Test Read operations"""
        print("\n=== Testing Read Operations ===")
        
        # Find one document
        try:
            doc = self.collection.find_one({"station_id": "TEST001"})
            print("\nFound single document:")
            pprint(doc)
        except Exception as e:
            print(f"Error finding single document: {e}")

        # Find multiple documents
        try:
            docs = self.collection.find({"operator": "Test Operator"})
            print("\nFound documents for Test Operator:")
            for doc in docs:
                pprint(doc)
        except Exception as e:
            print(f"Error finding multiple documents: {e}")

        # Count documents
        try:
            count = self.collection.count_documents({"operator": "Test Operator"})
            print(f"\nTotal test documents: {count}")
        except Exception as e:
            print(f"Error counting documents: {e}")

    def test_update_operations(self):
        """Test Update operations"""
        print("\n=== Testing Update Operations ===")
        
        # Update one document
        try:
            result = self.collection.update_one(
                {"station_id": "TEST001"},
                {"$set": {"status": "maintenance"}}
            )
            print(f"Modified {result.modified_count} document")
        except Exception as e:
            print(f"Error updating single document: {e}")

        # Update multiple documents
        try:
            result = self.collection.update_many(
                {"operator": "Test Operator"},
                {"$inc": {"charging_points": 1}}
            )
            print(f"Modified {result.modified_count} documents")
        except Exception as e:
            print(f"Error updating multiple documents: {e}")

    def test_delete_operations(self):
        """Test Delete operations"""
        print("\n=== Testing Delete Operations ===")
        
        # Delete one document
        try:
            result = self.collection.delete_one({"station_id": "TEST001"})
            print(f"Deleted {result.deleted_count} document")
        except Exception as e:
            print(f"Error deleting single document: {e}")

        # Delete multiple documents
        try:
            result = self.collection.delete_many({"operator": "Test Operator"})
            print(f"Deleted {result.deleted_count} documents")
        except Exception as e:
            print(f"Error deleting multiple documents: {e}")

    def run_all_tests(self):
        """Run all CRUD tests"""
        try:
            self.test_create_operations()
            self.test_read_operations()
            self.test_update_operations()
            self.test_delete_operations()
        except Exception as e:
            print(f"Error in test execution: {e}")
        finally:
            self.client.close()
            print("\nMongoDB connection closed")

if __name__ == "__main__":
    # Run tests
    tester = MongoDBCRUDTester()
    tester.run_all_tests()