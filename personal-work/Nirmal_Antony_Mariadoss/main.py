from fastapi import FastAPI, Request
import requests

app = FastAPI()

GOOGLE_API_KEY = "AIzaSyBttID79kCG9XQP1MO-7a1OOqG-PfpqBiY"
OCM_API_KEY = "d0fee2b1-2fa3-4725-ba42-d8073437d320"


def get_coordinates(city):
    # Append ", Victoria" to bias all searches to Victoria, AU
    city = f"{city.strip()}, Victoria"
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={city}&region=AU&key={GOOGLE_API_KEY}"
    response = requests.get(url).json()

    if response.get('status') == 'OK' and response.get('results'):
        loc = response['results'][0]['geometry']['location']
        return loc['lat'], loc['lng']

    return None, None

def get_chargers(lat, lon):
    url = "https://api.openchargemap.io/v3/poi/"
    params = {
        "output": "json",
        "latitude": lat,
        "longitude": lon,
        "maxresults": 3,
        "distance": 5,
        "distanceunit": "KM",
        "key": OCM_API_KEY
    }
    return requests.get(url, params=params).json()

def get_nearby_places(lat, lon, place_type="cafe"):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lon}",
        "radius": 500,
        "type": place_type,
        "key": GOOGLE_API_KEY
    }
    return requests.get(url, params=params).json().get("results", [])[:3]

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    intent = data['queryResult']['intent']['displayName']
    params = data['queryResult']['parameters']

    if intent == "FindCharger":
        city = params.get('geo-city', '').strip()
        lat, lon = get_coordinates(city)
        if not lat or not lon:
            return {"fulfillmentText": f"Sorry, I couldn't find the location '{city}'."}

        chargers = get_chargers(lat, lon)
        if not chargers:
            return {"fulfillmentText": f"No chargers found in {city}."}

        suggestions = []
        context_data = []
        for c in chargers:
            label = f"{c['AddressInfo']['Title']} – {c['AddressInfo'].get('AddressLine1', 'Unknown')}"
            suggestions.append(label)
            context_data.append({
                "label": label,
                "lat": c['AddressInfo']['Latitude'],
                "lon": c['AddressInfo']['Longitude']
            })

        return {
            "fulfillmentText": f"Here are chargers in {city}. Please select one:",
            "fulfillmentMessages": [{
                "quickReplies": {
                    "title": "Choose a station:",
                    "quickReplies": suggestions
                }
            }],
            "outputContexts": [{
                "name": f"{data['session']}/contexts/awaiting_selection",
                "lifespanCount": 5,
                "parameters": {
                    "chargers": context_data
                }
            }]
        }

    elif intent == "SelectCharger":
        user_input = data['queryResult']['queryText'].strip()
        context = next((ctx for ctx in data['queryResult']['outputContexts']
                        if 'awaiting_selection' in ctx['name']), {})
        charger_list = context.get("parameters", {}).get("chargers", [])
        selected = next((c for c in charger_list if c["label"].lower() == user_input.lower()), None)

        if not selected:
            return {"fulfillmentText": "Sorry, that charger wasn't recognized."}

        return {
            "fulfillmentText": "What would you like to see nearby? Cafés, restrooms, or convenience stores?",
            "fulfillmentMessages": [{
                "quickReplies": {
                    "title": "Choose a place type:",
                    "quickReplies": ["cafes", "restrooms", "convenience stores"]
                }
            }],
            "outputContexts": [{
                "name": f"{data['session']}/contexts/awaiting_amenity_type",
                "lifespanCount": 5,
                "parameters": {
                    "selected": selected
                }
            }]
        }

    elif intent == "SelectAmenityType":
        amenity = params.get("amenity_type", "").strip()
        context = next((ctx for ctx in data['queryResult']['outputContexts']
                        if 'awaiting_amenity_type' in ctx['name']), {})
        selected = context.get("parameters", {}).get("selected", {})
        lat = selected.get("lat")
        lon = selected.get("lon")
        label = selected.get("label")

        if not lat or not lon:
            return {"fulfillmentText": "Sorry, something went wrong with the location."}

        amenity_map = {
            "cafes": "cafe",
            "restrooms": "restroom",
            "convenience stores": "convenience_store"
        }

        place_type = amenity_map.get(amenity.lower(), "cafe")
        places = get_nearby_places(lat, lon, place_type)

        if not places:
            return {"fulfillmentText": f"No {amenity} found near {label}."}

        text = f"Here are some {amenity} near {label}:\n"
        for p in places:
            name = p.get('name')
            vicinity = p.get('vicinity', 'unknown location')
            text += f"- {name} ({vicinity})\n"

        return {"fulfillmentText": text}

    return {"fulfillmentText": "Sorry, I couldn't process your request."}
