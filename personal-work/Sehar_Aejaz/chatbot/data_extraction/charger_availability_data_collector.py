import find_station.py
import pandas as pd
import requests
import csv
import time
from datetime import datetime

def log_station_info(info, suburb_location):
    row = [
        datetime.now().isoformat(),
        info["Name"],
        info["Location"][0],
        info["Location"][1],
        info["Address"],
        round(info["Distance"], 2),
        round(info["ETA"], 2),
        suburb_location[1],  # Suburb latitude
        suburb_location[0]   # Suburb longitude
    ]
    # Print collected data
    print("Collected data:", row)

    with open(OUTPUT_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)



df = pd.read_csv('Co-oridnates.csv')  

# Convert latitude and longitude into tuples and store them in a list
coordinates_list = list(zip(df['longitude'], df['latitude']))

# CONFIG 
TOMTOM_API_KEY = "API KEY"
OUTPUT_FILE = "ml_ev_charging_dataset.csv"
COLLECTION_INTERVAL_SECONDS = 600  # Every 15 minutes

# INIT CSV 
with open(OUTPUT_FILE, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'Timestamp', 'Station_Name', 'Longitude', 'Latitude',
        'Address', 'Distance_km', 'ETA_min', 'Suburb_Location_Lat', 'Suburb_Location_Lon'
    ])


# Main Collection Loop 
print("Starting data collection...")

while True:
    for suburb_location in coordinates_list:
        try:
            print(f"Fetching data for suburb at {suburb_location}...")
            result = find_station(suburb_location)
            if result:
                log_station_info(result, suburb_location)
                print(f"Logged data for {suburb_location} at {datetime.now()}: {result['Name']}")
            else:
                print(f"No data for suburb {suburb_location}.")
        except Exception as e:
            print(f"Error fetching data for {suburb_location}: {e}")

    time.sleep(COLLECTION_INTERVAL_SECONDS)

# Interrupt the kernel when you want to stop data collection
