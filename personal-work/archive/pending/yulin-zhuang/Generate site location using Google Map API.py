import pandas as pd
import googlemaps
import time

# Load the CSV
df = pd.read_csv('EVCS_Usage_Three_Years.csv')

# Step 1: Collect unique site names
unique_sites = df['Site'].unique()

# Step 2: Set up Google Maps Geocoding client
gmaps = googlemaps.Client(key='AIzaSyCFd_5D3I1yhC8nguRzenQUg_NbGg_3rMU')

# Function to get latitude and longitude using Google Maps API
def get_coordinates_google(site):
    try:
        geocode_result = gmaps.geocode(f"{site}, Perth and Kinross, Scotland")
        if geocode_result:
            location = geocode_result[0]['geometry']['location']
            return location['lat'], location['lng']
        else:
            return None, None
    except Exception as e:
        print(f"Error fetching {site}: {e}")
        return None, None

# Step 3: Create a dictionary for site coordinates using Google Maps API
site_coordinates = {}
for site in unique_sites:
    if site != "***TEST SITE*** Charge Your Car HQ":  # Skip test site
        lat, lon = get_coordinates_google(site)
        site_coordinates[site] = (lat, lon)
        time.sleep(1)  # Pause to avoid exceeding the rate limit

# Step 4: Create latitude and longitude columns in the original dataframe
df['Latitude'] = df['Site'].map(lambda site: site_coordinates.get(site, (None, None))[0])
df['Longitude'] = df['Site'].map(lambda site: site_coordinates.get(site, (None, None))[1])

# Step 5: Remove rows where 'Site' is "***TEST SITE*** Charge Your Car HQ"
df = df[df['Site'] != "***TEST SITE*** Charge Your Car HQ"]

# Step 6: Save the updated dataframe to a new CSV
df.to_csv('EVCS_Usage_With_Google_Coordinates.csv', index=False)

print("Data cleaned and Google Maps coordinates added successfully!")
