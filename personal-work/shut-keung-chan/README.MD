# EV Route Planner (Intergrated Route Optimisation Function)

## Overview

The EV Route Planner is a Python function that helps electric vehicle (EV) drivers plan their routes, taking into account the vehicle's range and the need for charging stops. It uses the OpenRouteService API for route planning and geocoding, and the OpenChargeMap API for finding EV charging stations along the route.

## Features

- Calculate the optimal route between two locations
- Determine necessary charging stops based on the vehicle's range
- Display the route and charging stops on an interactive map
- Provide detailed information about the journey, including distance and estimated battery consumption

## Installation

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

### Required Libraries

Install the required libraries using pip:

```bash
pip install openrouteservice folium pandas geopy requests
```

### API Keys

You'll need to obtain API keys for the following services:

1. OpenRouteService: Sign up at [https://openrouteservice.org/](https://openrouteservice.org/)
2. OpenChargeMap: Sign up at [https://openchargemap.org/site/developerinfo](https://openchargemap.org/site/developerinfo)

## Usage

Since the EV Route Planner function is contained in a Jupyter notebook (`integrated_ev_route_planner.ipynb`), you have two main options for using it:

### Option 1: Use the function directly in the notebook

1. Download the `integrated_ev_route_planner.ipynb` file from this repository.
2. Open the notebook in Jupyter Lab or Jupyter Notebook.
3. Run all the cells in the notebook up to the function definition to ensure all necessary imports and the function itself are loaded.
4. In a new cell, you can now use the function:

```python
route_map, route_data, total_distance, total_battery_consumed = integrated_ev_route_planner(
    origin,
    destination,
    max_range,
    battery_size,
    openrouteservice_api_key,
    openchargemap_api_key
)
```

### Option 2: Copy the function to your own notebook or script

1. Open the `integrated_ev_route_planner.ipynb` file in a text editor or Jupyter environment.
2. Copy the entire `integrated_ev_route_planner` function, including all its nested functions.
3. Paste the function into your own Jupyter notebook or Python script.
4. Ensure you have all the necessary imports at the beginning of your notebook or script:

```python
import openrouteservice
import folium
import pandas as pd
from geopy.distance import geodesic
import json
import requests
```

5. You can now use the function in your notebook or script:

```python
route_map, route_data, total_distance, total_battery_consumed = integrated_ev_route_planner(
    origin,
    destination,
    max_range,
    battery_size,
    openrouteservice_api_key,
    openchargemap_api_key
)
```

## Function Details

### `integrated_ev_route_planner(origin, destination, max_range, battery_size, openrouteservice_api_key, openchargemap_api_key)`

#### Parameters:

- `origin` (str): The starting point of the journey (e.g., "Melbourne CBD, VIC, Australia")
- `destination` (str): The end point of the journey (e.g., "Geelong, VIC, Australia")
- `max_range` (float): The maximum range of the EV in kilometers
- `battery_size` (float): The battery capacity of the EV in kWh
- `openrouteservice_api_key` (str): Your OpenRouteService API key
- `openchargemap_api_key` (str): Your OpenChargeMap API key

#### Returns:

- `route_map` (folium.Map): An interactive map object showing the route and charging stops
- `route_data` (pandas.DataFrame): A dataframe containing detailed information about each point on the route
- `total_distance` (float): The total distance of the journey in kilometers
- `total_battery_consumed` (float): The estimated total battery consumption as a percentage

## Example

```python
origin = "Melbourne CBD, VIC, Australia"
destination = "Geelong, VIC, Australia"
max_range = 200  # km
battery_size = 100  # kWh
openrouteservice_api_key = "your_openrouteservice_api_key"
openchargemap_api_key = "your_openchargemap_api_key"

route_map, route_data, total_distance, total_battery_consumed = integrated_ev_route_planner(
    origin, destination, max_range, battery_size, openrouteservice_api_key, openchargemap_api_key
)

print(f"Total distance: {total_distance:.2f} km")
print(f"Total battery consumed: {total_battery_consumed:.2f}%")
print("\nRoute details:")
print(route_data)

# Display the map (in Jupyter Notebook or similar environment)
route_map
```

## Notes

- The function uses a buffer range of 50 km when searching for charging stations. This means it will start looking for a charging station when the remaining range is 50 km less than the maximum range.
- The function assumes a constant energy consumption rate based on the provided maximum range and battery size. In reality, energy consumption can vary based on factors such as driving style, terrain, and weather conditions.
- The accuracy of the results depends on the data provided by OpenRouteService and OpenChargeMap. Always verify the information and plan accordingly.
- The EV chargers information is based on the OpenChargeMap, for the EVAT mobile app, we will have our own database of EV charging stations.

## Troubleshooting

If you encounter any issues:

1. Ensure that your API keys are valid and have the necessary permissions.
2. Check your internet connection, as the function requires access to external APIs.
3. Verify that the origin and destination are valid and recognized by the geocoding service.
4. If no charging stations are found, try increasing the `max_range` or check if there are any charging stations along your route in the OpenChargeMap database.

## Contributing

Contributions to improve the EV Route Planner are welcome. Please feel free to submit pull requests or open issues to discuss potential improvements.

## License

[MIT License](https://opensource.org/licenses/MIT)