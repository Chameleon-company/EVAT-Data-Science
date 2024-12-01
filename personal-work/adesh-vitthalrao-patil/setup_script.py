import os
import sys
import json
from pathlib import Path

def create_project_structure():
    """Create the necessary project directories"""
    directories = [
        'data',
        'logs',
        'config',
        'notebooks',
        'scripts'
    ]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for directory in directories:
        dir_path = os.path.join(current_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def create_config():
    """Create the configuration file"""
    config = {
        "mongodb_uri": "mongodb+srv://EVAT:EVAT123@cluster0.5axoq.mongodb.net/",
        "database_name": "EVAT",  # Updated to your actual database name
        "collection_name": "charging_stations",  # Updated to your actual collection
        "export_path": "./data/",
        "refresh_interval": 300,
        "validation": {
            "lat_range": [-90, 90],
            "lon_range": [-180, 180],
            "required_fields": [
                "station_id",
                "latitude",
                "longitude",
                "timestamp"
            ]
        },
        "collections": {
            "charging_stations": {
                "document_count": 1600,
                "indexes": 2
            },
            "ev_vehicles": {
                "document_count": 200,
                "indexes": 1
            },
            "users": {
                "document_count": 2,
                "indexes": 1
            }
        }
    }
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config', 'config.json')
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Created config file: {config_path}")

def setup_virtual_env():
    """Print instructions for virtual environment setup"""
    instructions = """
    Please run the following commands in your terminal:

    # Create virtual environment
    python -m venv heatmap_env

    # Activate virtual environment
    # On macOS/Linux:
    source heatmap_env/bin/activate
    # On Windows:
    # heatmap_env\\Scripts\\activate

    # Install required packages
    pip install pymongo pandas numpy matplotlib seaborn jupyter
    pip install python-dotenv schedule
    """
    print(instructions)

def main():
    try:
        print("Setting up project structure...")
        create_project_structure()
        
        print("\nCreating configuration file...")
        create_config()
        
        print("\nProject setup complete!")
        print("\nVirtual Environment Setup Instructions:")
        setup_virtual_env()
        
    except Exception as e:
        print(f"Error during setup: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()