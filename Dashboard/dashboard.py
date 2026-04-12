import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import folium

st.set_page_config(layout="wide")

st.title("EV Charging Station Dashboard")

# --- Initialize Session State ---
if 'stations_df' not in st.session_state:
    st.session_state.stations_df = pd.DataFrame()
if 'center_lat' not in st.session_state:
    st.session_state.center_lat = -25.2744
if 'center_lon' not in st.session_state:
    st.session_state.center_lon = 133.7751
if 'zoom_start' not in st.session_state:
    st.session_state.zoom_start = 4

# --- Postcode Input ---
st.sidebar.header("Location")
postcode = st.sidebar.text_input("Enter an Australian Postcode", "3000")
radius = 5  # 5 km

@st.cache_data
def geocode_postcode(postcode):
    """Geocodes a postcode to get latitude and longitude."""
    try:
        geolocator = Nominatim(user_agent="ev_dashboard")
        location = geolocator.geocode(f"{postcode}, Australia")
        if location:
            return location.latitude, location.longitude
        else:
            st.sidebar.error("Could not find location for the entered postcode.")
            return None, None
    except Exception as e:
        st.sidebar.error(f"Geocoding error: {e}")
        return None, None

# --- Data Loading ---
@st.cache_data
def load_station_data():
    """Loads charging station data, renaming columns for map compatibility."""
    try:
        data = pd.read_csv('EVAT.chargers.csv')
        data.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)
        
        # Filter for stations within Australia
        australia_bounds = {
            "min_lat": -44, "max_lat": -10,
            "min_lon": 112, "max_lon": 154
        }
        data = data[
            (data['lat'].between(australia_bounds['min_lat'], australia_bounds['max_lat'])) &
            (data['lon'].between(australia_bounds['min_lon'], australia_bounds['max_lon']))
        ]
        return data
    except FileNotFoundError:
        st.error("Charging station data not found. Please check the file path.")
        return pd.DataFrame()

all_stations_df = load_station_data()

if st.sidebar.button("Find Stations"):
    lat, lon = geocode_postcode(postcode)
    if lat is not None and lon is not None:
        st.session_state.center_lat, st.session_state.center_lon = lat, lon
        st.session_state.zoom_start = 12
        
        st.session_state.stations_df = all_stations_df[
            all_stations_df.apply(
                lambda row: geodesic((lat, lon), (row['lat'], row['lon'])).km <= radius,
                axis=1
            )
        ].copy()
        
        if st.session_state.stations_df.empty:
            st.warning("No charging stations found within 5km of the entered postcode.")
        else:
            st.session_state.stations_df['color'] = '#0000FF' # Blue for initial view

# --- Congestion Prediction ---
if not st.session_state.stations_df.empty:
    st.sidebar.header("Congestion Prediction")
    prediction_date = st.sidebar.date_input("Select a date", datetime.now())
    prediction_time = st.sidebar.time_input("Select a time", datetime.now().time())

    if st.sidebar.button("Predict Congestion for All Stations"):
        api_url = "http://localhost:8000/predict"
        
        progress_bar = st.sidebar.progress(0)
        total_stations = len(st.session_state.stations_df)
        
        for i, (index, row) in enumerate(st.session_state.stations_df.iterrows()):
            station_id = row['_id']
            try:
                response = requests.post(api_url, json={"station_id": station_id})
                response.raise_for_status()
                
                prediction_data = response.json()
                congestion = prediction_data.get("congestion_level", "Unknown")

                if congestion == 'low':
                    color = '#00FF00'  # Green
                elif congestion == 'medium':
                    color = '#FFFF00'  # Yellow
                elif congestion == 'high':
                    color = '#FF0000'  # Red
                else:
                    color = '#808080'  # Grey for unknown
                
                st.session_state.stations_df.loc[index, 'color'] = color

            except requests.exceptions.RequestException as e:
                st.sidebar.warning(f"Could not predict for {station_id}: {e}")
            
            progress_bar.progress((i + 1) / total_stations)

# --- Legend ---
st.sidebar.header("Legend")
st.sidebar.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 5px;">
        <div style="width: 20px; height: 20px; background-color: #00FF00; border-radius: 50%; margin-right: 10px;"></div>
        <span>Low Congestion</span>
    </div>
    <div style="display: flex; align-items: center; margin-bottom: 5px;">
        <div style="width: 20px; height: 20px; background-color: #FFFF00; border-radius: 50%; margin-right: 10px;"></div>
        <span>Medium Congestion</span>
    </div>
    <div style="display: flex; align-items: center; margin-bottom: 5px;">
        <div style="width: 20px; height: 20px; background-color: #FF0000; border-radius: 50%; margin-right: 10px;"></div>
        <span>High Congestion</span>
    </div>
    <div style="display: flex; align-items: center; margin-bottom: 5px;">
        <div style="width: 20px; height: 20px; background-color: #0000FF; border-radius: 50%; margin-right: 10px;"></div>
        <span>Initial Station</span>
    </div>
    <div style="display: flex; align-items: center; margin-bottom: 5px;">
        <div style="width: 20px; height: 20px; background-color: #808080; border-radius: 50%; margin-right: 10px;"></div>
        <span>Unknown</span>
    </div>
""", unsafe_allow_html=True)

# --- Map Display ---
if not st.session_state.stations_df.empty:
    st.map(st.session_state.stations_df, color='color')
else:
    st.info("Enter a postcode and click 'Find Stations' to begin.")

