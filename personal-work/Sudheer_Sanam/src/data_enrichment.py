import pandas as pd
import os
from pathlib import Path
import requests
import time

# -----------------------
# Paths
# -----------------------
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = CURRENT_DIR / ".." / "data" / "raw"
PROCESSED_PATH = CURRENT_DIR / ".." / "data" / "processed"
INTERIM_PATH = CURRENT_DIR / ".." / "data" / "interim"

# Open Charge Map Free API
OCM_API_URL = "https://api.openchargemap.io/v3/poi/"
OCM_API_KEY = "OCM-API-TEST"  # Free test key from Open Charge Map

def load_sessions_data():
    """Load raw session data for enrichment."""
    raw_file = RAW_PATH / "station_data_dataverse.csv"
    if raw_file.exists():
        return pd.read_csv(raw_file)
    else:
        raise FileNotFoundError("❌ station_data_dataverse.csv not found in raw folder.")

def fetch_station_data(station_id):
    """Fetch station details from Open Charge Map API."""
    try:
        params = {
            "key": OCM_API_KEY,
            "chargepointid": station_id,
            "output": "json"
        }
        response = requests.get(OCM_API_URL, params=params, timeout=10)
        if response.status_code == 200 and response.json():
            data = response.json()[0]
            address_info = data.get("AddressInfo", {})
            return {
                "station_id": station_id,
                "latitude": address_info.get("Latitude", None),
                "longitude": address_info.get("Longitude", None),
                "city_enriched": address_info.get("Town", None),
                "title": address_info.get("Title", None)
            }
    except Exception as e:
        print(f"⚠️ Error fetching data for station {station_id}: {e}")
    return {
        "station_id": station_id,
        "latitude": None,
        "longitude": None,
        "city_enriched": None,
        "title": None
    }

def enrich_sessions_with_ocm():
    """Main enrichment function."""
    df_sessions = load_sessions_data()
    
    # Ensure stationId exists
    if "stationId" not in df_sessions.columns:
        raise ValueError("❌ 'stationId' column missing in session data.")

    # Deduplicate station IDs to minimize API calls
    unique_ids = df_sessions["stationId"].dropna().unique()
    print(f"🔄 Fetching details for {len(unique_ids)} unique stations from OCM API...")

    enriched_data = []
    for sid in unique_ids:
        enriched_data.append(fetch_station_data(sid))
        time.sleep(1)  # Respect OCM free API rate limit

    # Convert to DataFrame
    df_enriched = pd.DataFrame(enriched_data)

    # Merge with session data
    df_merged = df_sessions.merge(df_enriched, how="left", left_on="stationId", right_on="station_id")
    df_merged.drop(columns=["station_id"], inplace=True)

    # Save to interim folder
    INTERIM_PATH.mkdir(parents=True, exist_ok=True)
    output_file = INTERIM_PATH / "sessions_enriched_ocm.csv"
    df_merged.to_csv(output_file, index=False)
    print(f"✅ Enriched session data saved to {output_file}")

if __name__ == "__main__":
    enrich_sessions_with_ocm()
