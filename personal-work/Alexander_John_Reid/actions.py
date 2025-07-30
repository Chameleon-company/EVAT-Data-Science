# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

import arrow
import dateparser
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Text, Dict, List
from backend.find_station import find_station
from backend.find_station import get_route_details
from backend.find_station import get_charging_station_availability
import pandas as pd


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []



class ActionGetNearestStation(Action):

    def name(self) -> Text:
        return "Action_Get_Nearest_Station"  

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Retrieve the latest message's metadata, it shoiuld have latitude and longitude
        metadata = tracker.latest_message.get('metadata', {})

        latitude = metadata.get("lat")
        longitude = metadata.get("lon")

        if not latitude and not longitude:
            latitude = -37.85580046992546
            longitude = 145.08025857057336
            # manual location due to geocoder not getting my location -37.85580046992546, 145.08025857057336

        
        
        user_location = (longitude, latitude)
        
        if latitude and longitude:

            dispatcher.utter_message(text="I am fetching the nearest station information.")
            result = find_station(user_location)

            if result:            
                
                dispatcher.utter_message(text=f"Your closest charging station is {result['Name']}")
                dispatcher.utter_message(text=f"It is located at {result['Address']}")
                dispatcher.utter_message(text=f"It is about {result['Distance']} KM away")
                dispatcher.utter_message(text=f"This will take you about {result['ETA']} minutes")
                dispatcher.utter_message("Would you like directions?")
                
                return [SlotSet("Charger Name", result['Address'])] 
            else:
                dispatcher.utter_message("Sorry, no charger is currently available in your location")
        else:
            # No location available
            dispatcher.utter_message("Sorry, I couldn't retrieve your location.")
            return []
    

class ActionToChargingStation(Action):

    def name(self) -> Text:
        return "Action_To_Charging_Station" 
    
    

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        station = list(tracker.get_latest_entity_values("place"))
        metadata = tracker.latest_message.get('metadata', {})

        latitude = metadata.get("lat")
        longitude = metadata.get("lon")

        if not latitude and not longitude:
            latitude = -37.85580046992546
            longitude = 145.08025857057336
            # manual location due to geocoder not getting my location -37.85580046992546, 145.08025857057336


        df = pd.read_csv("datasets/Co-oridnates.csv")
        df = df.astype(str)
        charging_station = df[df["suburb"].str.strip().str.lower() == station[0]]
        station_info = charging_station.iloc[0]
        
        destination = (station_info['longitude'], station_info['latitude'])
       
        
    
        user_location = (longitude, latitude,)
        if station:


            station_name = station[0]  
            dispatcher.utter_message(text=f"I understand. Taking you to the {station_name} charging station.")
            result = get_route_details(user_location, destination)

            if result:
                    directions = result["instructions"]
                    for step in directions:
                     dispatcher.utter_message(text=(step))
            else:
                print("Could not retrieve route directions.")
            
            

        else:
            
            dispatcher.utter_message(text="i did not extract a location")
        











        return []
    

       
    
    



class ActionHowLongToCharge(Action): 

    def name(self) -> Text:
        return "Action_How_Long_To_Charge" 
    

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="It takes 20hr on slow, 12hr on 7kw fast, 7hr on 22kw fast, 1 hr on 43-50kw rapid, 30min on 150kw charge")
        
        
        return []
    
class ActionDistanceICanGo(Action):

    def name(self) -> Text:
        return "Action_Distence_I_Can_Go" 
    

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="depending on your car and driving style. you can expect between. 320 to 480km on a full charge")
        
        
        return []

    

class ActionFilterStations(Action):

    def name(self) -> Text:
        return "Action_Filter_Stations"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        filter_preferences = tracker.get_slot("filter_preferences")

        if filter_preferences:
            dispatcher.utter_message(text=f"Filtering stations based on your preference: {filter_preferences}")
        else:
            dispatcher.utter_message(text="No filter preferences found. Showing all stations.")
        
        return []





class ActionTrafficInfo(Action):

    def name(self) -> Text:
        return "Action_Traffic_Info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Sorry, I couldn't fetch traffic information at the moment. Please try again later.")
        return []






class ActionDefaultFallback(Action):

    def name(self) -> Text:
        return "action_default_fallback"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Sorry, I didn't get that. Can you rephrase?")
        return []
    

class ActionChargerInfo(Action):

    def name(self) -> Text:
        return "Action_charger_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        df = pd.read_csv("datasets/charger_info_mel.csv")
        df = df.astype(str)
        address = tracker.get_slot("Charger Name")
        
        
        
        dispatcher.utter_message(text="Sure, here are important information about the charging station.")
        
        
        charging_station = df[df["Address"].str.strip().str.lower() == address.strip().lower()]
        print(address)
        print(address)
        print(address)
        print(charging_station)
            
        if not charging_station.empty:
            station_info = charging_station.iloc[0]  

            dispatcher.utter_message(text=f"Suburb: {station_info['City']}")
            dispatcher.utter_message(text=f"Power output: {station_info['Power (kW)']} kW")
            dispatcher.utter_message(text=f"Usage costs: {station_info['Usage Cost']}")
            dispatcher.utter_message(text=f"Total charges: {station_info['Number of Points']}")
            dispatcher.utter_message(text=f"Connection type: {station_info['Connection Types']}")
            #get_charging_station_availability()
        else:
            dispatcher.utter_message(text=f"sorry we are unable to get details from the {address} charging station")
            
        
        return []
    

class ActionDefaultFallback(Action):

    def name(self) -> Text:
        return "Get_Directions_action"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        

        metadata = tracker.latest_message.get('metadata', {})

        latitude = metadata.get("lat")
        longitude = metadata.get("lon")

        if not latitude and not longitude:
            latitude = -37.85580046992546
            longitude = 145.08025857057336
            # manual location due to geocoder not getting my location -37.85580046992546, 145.08025857057336

        
        
        user_location = (longitude, latitude)

        result = find_station(user_location)
        dispatcher.utter_message(text=f"Your closest charging station is {result['Instructions']}")
        
        if "Instructions" in result:
            print("\nInstructions:")
            for i, instruction in enumerate(result["Instructions"], start=1):
                dispatcher.utter_message(text=(f"{i}. {instruction}"))


        print(result)

        return []