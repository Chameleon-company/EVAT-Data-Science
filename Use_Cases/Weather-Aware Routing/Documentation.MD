<aside>
📘

**Developer Onboarding Overview**

Combined System: Sprint 4 Routing Engine + Flask Prediction API

*April 2026*

</aside>

# 1. What this project is trying to do

EVAT (Electric Vehicle Adoption Tools) is a data science project that aims to help EV drivers make smarter routing decisions by combining three types of real-world data: weather conditions, traffic congestion, and EV charger availability. The end goal of this use case is a web application where a user enters a start and destination, and the system returns the optimal route — one that accounts for battery range limitations, charging stop locations, and how weather degrades range.

Two sub-teams have been working on complementary pieces of this system. They have not yet been connected together, which is the main integration task ahead.

![image.png](attachment:7a18e742-eeff-4e09-9113-65f82e53ea24:image.png)

# 2. The two codebases

![image.png](attachment:51b482a8-e908-4664-8bc4-e12bb72b890e:image.png)

## 2.1 Sprint 4 — routing engine (Python)

This codebase implements the core routing algorithm. It builds a graph of EV charging stations and finds the lowest-cost path between a start and goal node using Dijkstra's algorithm. Weather and traffic scores are applied as cost multipliers on each edge, so the algorithm naturally prefers routes with better conditions.

| File | What it does |
| --- | --- |
| engine_sprint_4.py | Main routing pipeline: loads data, builds graph, runs Dijkstra, returns route summary |
| adapters_sprint_4.py | Stub for live weather/traffic API calls — currently returns static defaults only |
| utils_sprint_4.py | Haversine distance calculation and min-max normalisation helpers |
| ui_app_sprint_4.py | Streamlit frontend — shows a Folium map with the routed path |

The routing pipeline runs in five steps:

- `load_dataset()` reads the CSV and computes weather and traffic scores for each station
- `build_nodes()` extracts station coordinates and scores into a clean node table
- `build_edges()` connects each node to its K nearest neighbours within EV range
- `build_costs()` multiplies base travel time by weather and traffic penalties
- `dijkstra_with_stop_penalty()` finds the shortest cost path, adding a charge time penalty at each intermediate stop

## 2.2 Flask prediction API (Python + React)

This codebase serves a trained machine learning model via a REST API, with a React frontend that lets the user click on a Google Maps location and receive a prediction. The model is a RandomForestRegressor trained on six features derived from EV station proximity, road geometry, and historical weather.

| File | What it does |
| --- | --- |
| api.py | Flask server with /predict endpoint — looks up nearest road, EV station, and weather station for a coordinate, then runs the model |
| ev_model.pkl | Trained RandomForest model — 100 estimators, 6 features |
| apiClient.ts | TypeScript client — validates inputs and calls the Flask API |
| MapComponent.tsx | React component — Google Maps click handler, calls API, displays prediction result card |
| useEVPrediction.ts | React hook wrapping the API call with loading/error state management |

When a user clicks the map, the frontend sends the coordinates to the Flask API. The API performs three spatial lookups — nearest road segment, nearest EV station, nearest weather station — then assembles a feature vector and calls `model.predict()`. The result is returned as JSON and displayed on the map.

# 3. How they should connect together

The two codebases solve complementary problems. The routing engine finds the best path through the charger network. The prediction model estimates energy consumption at a given point. In a fully integrated system, the routing engine would call the prediction model for each edge in the graph to estimate kWh cost, not just travel time. This creates a genuinely energy-aware route.

The target integration flow is:

- User enters start address and destination in the React frontend
- Frontend geocodes both addresses to coordinates using Google Maps Geocoding API
- Frontend sends a single POST request to a unified API with start_lat, start_lon, goal_lat, goal_lon, and battery_pct
- The unified API runs the routing engine to find candidate paths through the charger graph
- For each edge in each candidate path, it calls the energy model to estimate kWh consumed
- The path with the lowest total energy cost that fits within the battery range is returned
- The frontend renders the route on the map with charger stops marked and per-segment energy shown

This integration does not currently exist. The two APIs run on separate ports (8000 and 5001) and have no shared data contract. Building it requires agreeing on a unified request/response schema and combining them under a single FastAPI server.

# 4. Problems and gaps

## 4.1 Critical

**Wrong geographic data.** The EV station data in EV_stations_data.csv has coordinates in Victoria, BC, Canada (lat ~48.4, lon ~-123.3). The React frontend is centred on Melbourne, Australia. Every spatial lookup and model prediction in the Flask API is therefore geographically meaningless for Australian users. This data must be replaced with Melbourne EV station data before any other work on the prediction API is worth doing.

## 4.2 High severity

**Unknown model target.** The training notebook that produced ev_model.pkl is not in the repository. The UI labels the model output as 'kWh' but the features (road shape length, EV station proximity) suggest it may be predicting traffic volume, not energy. This must be clarified before the model is used in a real product.

**Two codebases not connected.** The routing engine (Sprint 4) and the prediction API operate as completely independent systems. There is no end-to-end path from user input to a routed, energy-estimated journey.

**ML model uses fabricated target variable.** The Random Forest in the Sprint 4 notebooks is trained to predict a Weather Sensitivity Score that was computed by hand from temperature and precipitation flags. Because the input features (TMAX, TMIN, PRCP) directly determine the target by the same formula, the model achieves R² ≈ 1.0 — not because it has learned anything useful, but because it is recovering a deterministic function it was given. The R² result is not a valid performance measure here.

## 4.3 Medium severity

- Static 2023 weather data — the API always uses annual averages regardless of current season or real-time conditions
- No spatial index — api.py scans all geometries on every request using linear search (O(n)); should use geopandas sindex or scipy KDTree
- Graph is too small — Sprint 4 routes across only 28 stations; a production Melbourne graph needs hundreds of Open Charge Map points
- Node indices as routing input — users cannot specify start/goal by address, only by integer node index in the CSV

## 4.4 Low severity

- Flask running with debug=True — must be False in any deployed environment
- Frontend hardcodes year=2026 — no user input for this field
- Google Maps API key not documented in env setup instructions
- ts hook imported but unused — MapComponent.tsx calls the API client directly

# 5. Ideal target architecture

The following describes what the complete, production-ready system should look like. This is the target to build toward.

## 5.1 Data layer (fix first)

- Replace EV station data with Melbourne data from Open Charge Map API (free, REST, live availability)
- Fetch live weather from Tomorrow.io or OpenWeatherMap along route waypoints at planning time
- Add elevation data (Google Elevation API or SRTM) to account for hill-based range loss

## 5.2 Routing engine (improve)

- Replace Dijkstra with A* using geographic heuristic for faster convergence on large graphs
- Switch edge costs from time-based to energy-based (kWh) so battery constraint can be enforced
- Use a real battery model: consumption increases ~30% below 10°C, scales with drag at speed, increases on uphill gradients
- Build spatial index with scipy KDTree to handle hundreds of nodes efficiently
- Accept start/goal as coordinates or address strings, not integer indices

## 5.3 Prediction model (retrain)

- Obtain or create a Melbourne dataset with known energy consumption per trip segment
- Retrain the RandomForest on real features: trip distance, elevation change, temperature, precipitation, speed
- Define the target variable clearly before training — it should be kWh consumed per km, not a synthetic score
- Validate on a held-out test set with genuine geographic diversity

## 5.4 Unified API (build)

- Consolidate Sprint 4 engine and Flask API into a single FastAPI server
- Expose three endpoints: POST /route (start + goal + battery), GET /stations, GET /health
- /route internally calls the routing engine, then for each route edge calls the energy model
- Run behind gunicorn or uvicorn in production, never Flask dev server

## 5.5 Frontend (extend)

- Add destination marker and address geocoding to complement the existing start location click
- Display route as a polyline with charger stops marked distinctly
- Show per-segment energy cost and total kWh for the journey
- Add battery level input so the system can warn if a charging stop is needed

# 6. Recommended external APIs

| **API** | **Purpose** | **Cost** | **Notes** |
| --- | --- | --- | --- |
| Google Maps Directions | Road-accurate routing between addresses | Paid, free tier | Required for geocoding and real road paths |
| Open Charge Map | Live EV charger locations and availability | Free | Replaces the Canadian station CSV |
| Tomorrow.io | Live weather along route waypoints | Free tier available | Replaces static 2023 CSV |
| Google Elevation API | Elevation profile for energy modelling | Paid, free tier | Needed for uphill/downhill range adjustment |
| NREL TRIP model | Physics-based EV range estimation | Free (US Gov) | Can seed training data for the ML model |

# 7. Getting started as a new developer

Follow this order to get the existing code running locally before starting on improvements.

### Step 1 — Sprint 4 routing engine

- `pip install fastapi uvicorn pandas numpy scikit-learn geopy`
- Place `combined_data.csv` in the evat-sprint4/data/ folder
- Run: `uvicorn api.main:app --host 127.0.0.1 --port 8000` from the evat-sprint4/ directory
- Visit `http://127.0.0.1:8000/docs` to test the /route endpoint interactively

### Step 2 — Flask prediction API

- pip install flask flask-cors pandas geopandas shapely joblib scikit-learn
- Ensure EV`_stations_data.csv`, `weather_by_station_2023.csv`, and `ev_model.pkl` are in the same folder as `api.py`
- Run: `python api.py`
- Run: `python test_api.py` to verify the /predict endpoint responds correctly

### Step 3 — React frontend

- `npm install` inside the frontend directory
- Create .env with `REACT_APP_GOOGLE_MAPS_API_KEY=your_key` and `REACT_APP_API_URL=http://localhost:5001`
- Run: `npm start`
- Click on the Melbourne map area — the prediction card should appear in the bottom-right corner

### Step 4 — First improvement task

The highest-value first task is replacing the Canadian EV station data with Melbourne data from Open Charge Map. This alone makes the prediction API geographically valid and unblocks all further work on the energy model.

*End of document*