import pandas as pd
import numpy as np
from pymongo import MongoClient
import json
import logging
import os
from datetime import datetime

class DataValidator:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config', 'config.json')
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.setup_logging()
        
    def setup_logging(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(current_dir, 'logs', 'validation.log')
        
        self.logger = logging.getLogger('DataValidator')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)
    
    def validate_coordinates(self, df):
        """Validate latitude and longitude ranges"""
        # First check if columns exist
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            self.logger.warning("Latitude or longitude columns missing")
            return df
            
        lat_mask = df['latitude'].between(*self.config['validation']['lat_range'])
        lon_mask = df['longitude'].between(*self.config['validation']['lon_range'])
        
        invalid_coords = df[~(lat_mask & lon_mask)]
        if not invalid_coords.empty:
            self.logger.warning(f"Found {len(invalid_coords)} records with invalid coordinates")
            return df[lat_mask & lon_mask]
        return df
    
    def validate_timestamps(self, df):
        """Validate and standardize timestamps"""
        try:
            if 'timestamp' not in df.columns:
                self.logger.warning("Timestamp column missing")
                return df
                
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            self.logger.error(f"Timestamp validation failed: {str(e)}")
            raise
    
    def print_data_summary(self, df):
        """Print summary of the data"""
        print("\nData Summary:")
        print(f"Total records: {len(df)}")
        print("\nColumns present:")
        for col in df.columns:
            print(f"- {col}")
        print("\nSample record:")
        print(df.iloc[0].to_dict())
        
    def validate_dataset(self, df):
        """Run all validations"""
        try:
            self.logger.info("Starting data validation")
            
            # Print initial data summary
            self.print_data_summary(df)
            
            # Run validations
            df = self.validate_coordinates(df)
            df = self.validate_timestamps(df)
            
            self.logger.info(f"Validation complete. Final dataset shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise

def main():
    """Test the validation pipeline"""
    validator = DataValidator()
    
    try:
        # Connect to MongoDB
        client = MongoClient(validator.config['mongodb_uri'])
        db = client[validator.config['database_name']]
        collection = db[validator.config['collection_name']]
        
        print("Connected to MongoDB successfully")
        print(f"Database: {validator.config['database_name']}")
        print(f"Collection: {validator.config['collection_name']}")
        
        # Get data
        data = list(collection.find())
        if not data:
            print("No data found in the collection")
            return
            
        print(f"Retrieved {len(data)} records from MongoDB")
        
        df = pd.DataFrame(data)
        
        # Validate
        clean_df = validator.validate_dataset(df)
        
        # Create output directory if it doesn't exist
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save validated data
        output_path = os.path.join(output_dir, f"validated_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
        clean_df.to_csv(output_path, index=False)
        
        print(f"\nValidation complete. Data saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()