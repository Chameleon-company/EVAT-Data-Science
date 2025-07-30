import requests
import pandas as pd

url = "https://api.openchargemap.io/v3/poi/"

params = {
    "output": "json",
    "latitude": -37.8136,  # Melbourne's latitude
    "longitude": 144.9631,  # Melbourne's longitude
    "distance": 10000,  # Search radius in kilometers
    "maxresults": 260,  # Max number of results to return
    "compact": True,
    "key": "dcfb7363-bec6-4a7c-84b4-681a8082da7e"  
}

response = requests.get(url, params=params)
data = response.json()

for poi in data:
    address = poi.get("AddressInfo", {})
    print(f"{address.get('Title')} - {address.get('AddressLine1')}, {address.get('Town')}, {address.get('StateOrProvince')}, {address.get('Postcode')}")



# Create an empty list to hold the charger info
charger_info = []

    # Loop through the data and extract charger address details
for poi in data:
    address_info = poi.get("AddressInfo", {})
    connections = poi.get("Connections", [])
        
        # Initialize variables to store data
    power_list = []  # To store power values separately
    charger_types = []  # To store charger types
    cost = poi.get("UsageCost", "N/A")
    charge_points = poi.get("NumberOfPoints", "N/A")
        # Process connections data
    for conn in connections:
            # Add power of each connection to the power_list
        power_list.append(conn.get('PowerKW', 'N/A'))

            # Assume the charger type is inferred from ConnectionTypeID or level 
        charger_types.append(conn.get('ConnectionTypeID', 'N/A'))

        # Add each charging station's info to the list
    charger_info.append({
        "Charger Name": address_info.get('Title', 'N/A'),
        "Address": address_info.get('AddressLine1', 'N/A'),
        "City": address_info.get('Town', 'N/A'),
        "State": address_info.get('StateOrProvince', 'N/A'),
        "Postal Code": address_info.get('Postcode', 'N/A'),
        "Power (kW)": ", ".join(str(Power) for power in power_list),  # Join power ratings by commas
        "Usage Cost": cost,
        "Number of Points": charge_points,
        "Connection Types": ", ".join(str(ct) for ct in charger_types)
        })

    # Create a pandas DataFrame from the list
df = pd.DataFrame(charger_info)
df.head()

# Save to a csv file
df.to_csv('charger_info_mel.csv', index=False)
