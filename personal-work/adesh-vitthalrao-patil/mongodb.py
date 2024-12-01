import pymongo
from pymongo import MongoClient
import pandas as pd
from datetime import datetime

class MongoDBLoader:
    def __init__(self):
        """Initialize MongoDB connection for EV database"""
        USERNAME = "adeshpatil0907"
        PASSWORD = "pXuwLJEt9YcdEQef"
        CONNECTION_STRING = f"mongodb+srv://{USERNAME}:{PASSWORD}@ev.l2r4t.mongodb.net/"
        
        try:
            self.client = MongoClient(CONNECTION_STRING)
            print("MongoDB connection initialized")
        except Exception as e:
            print(f"Error initializing MongoDB connection: {e}")
            raise
    
    def test_connection(self):
        try:
            self.client.admin.command('ismaster')
            print("Successfully connected to MongoDB!")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def load_dataframe_to_mongodb(self, df, database, collection):
        try:
            # Convert DataFrame to list of dictionaries
            records = df.to_dict(orient='records')
            
            # Get database and collection
            db = self.client[database]
            coll = db[collection]
            
            # Insert the records
            result = coll.insert_many(records)
            print(f"Successfully inserted {len(result.inserted_ids)} documents")
            return len(result.inserted_ids)
        except Exception as e:
            print(f"Error loading data to MongoDB: {e}")
            return 0

    def close_connection(self):
        try:
            self.client.close()
            print("MongoDB connection closed")
        except Exception as e:
            print(f"Error closing connection: {e}")

if __name__ == "__main__":
    try:
        # Load your existing CSV file
        file_path = "/Users/adeshpatil/heatmap_project/data/cleaned_charging_stationss.csv"
        print(f"\nLoading data from: {file_path}")
        df = pd.read_csv(file_path)
        
        print(f"\nSuccessfully loaded {len(df)} records from CSV")
        print("\nColumns in dataset:")
        for col in df.columns:
            print(f"- {col}")
        
        # Initialize MongoDB connection
        loader = MongoDBLoader()
        
        # Test connection
        if loader.test_connection():
            # Load data into MongoDB
            inserted_count = loader.load_dataframe_to_mongodb(
                df=df,
                database="EV_charging_station",
                collection="charging_stations"
            )
            
            print(f"\nSummary:")
            print(f"Total records processed: {len(df)}")
            print(f"Records inserted: {inserted_count}")
            
            # Display a sample document
            if inserted_count > 0:
                db = loader.client["EV_charging_station"]
                coll = db["charging_stations"]
                print("\nSample document from MongoDB:")
                print(coll.find_one())
        
        # Close connection
        loader.close_connection()
        
    except FileNotFoundError:
        print(f"Error: Could not find the CSV file at {file_path}")
        print("Please check if the file path is correct")
    except Exception as e:
        print(f"Error in data loading process: {e}")