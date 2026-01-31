#!/usr/bin/env python3
import requests
import json
import sys

API_URL = "http://localhost:5001"

def test_root():
    print("Testing root endpoint (/)...")
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            print(f"Success: {response.json()}")
            return True
        else:
            print(f"Failed: HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("Cannot connect to server. Please ensure the server is running.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict():
    print("\nTesting prediction endpoint (/predict)...")
    test_data = {
        "year": 2023,
        "start_lat": -37.8136,
        "start_lon": 144.9631
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("Success")
            print(f"   prediction: {result.get('prediction')}")
            print(f"   dist_to_nearest_ev_m: {result.get('dist_to_nearest_ev_m'):.2f}")
            print(f"   ev_within_500m: {result.get('ev_within_500m')}")
            return True
        else:
            print(f"Failed: HTTP {response.status_code}")
            print(f"   Error message: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("Cannot connect to server.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("API test script")
    print("=" * 50)
    
    if not test_root():
        print("\nServer does not seem to be running.")
        sys.exit(1)

    test_predict()
    
    print("\n" + "=" * 50)
    print("Done.")
    print("=" * 50)
