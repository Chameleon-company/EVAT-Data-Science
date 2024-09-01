class UserService:
    def __init__(self, user_model, station_model):
        self.user_model = user_model
        self.station_model = station_model

    def add_user(self, username, email):
        if not username or not email:
            raise ValueError("Username and email are required")

        return self.user_model.insert(username, email)

    def get_users(self):
        users = self.user_model.find_all()
        outcome = []
        for user in users:
            favorite_station_ids = user.get("favorite_stations", [])
            favorite_station_names = []

            for station_id in favorite_station_ids:
                station = self.station_model.find_by_id(station_id)
                if station:
                    favorite_station_names.append(station.get("station_name"))

            user_with_stations = {
                "user_name": user.get("user_name"),
                "email": user.get("email"),
                "favorite_stations": favorite_station_names
            }
            outcome.append(user_with_stations)
        return outcome

    def remove_user(self, user_id):
        return self.user_model.delete(user_id)

    def add_favorite_station(self, user_id, station_name):
        user = self.user_model.find_by_id(user_id)
        if not user:
            raise ValueError("User not found")

        station = self.station_model.find_by_name(station_name)
        if not station:
            raise ValueError("Station not found")

        if station['_id'] in user['favorite_stations']:
            raise ValueError("Station is already a favorite")

        return self.user_model.add_favorite_station(user_id, str(station['_id']))

    def remove_favorite_station(self, user_id, station_name):
        user = self.user_model.find_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        station = self.station_model.find_by_name(station_name)
        if not station:
            raise ValueError("Station not found")

        if station['_id'] not in user['favorite_stations']:
            raise ValueError("Station is not a favorite")

        return self.user_model.remove_favorite_station(user_id, station_name)

    def get_favorite_stations(self, user_id):
        user = self.user_model.find_by_id(user_id)
        if not user:
            raise ValueError("User not found")

        favorite_stations = []
        for station_id in user['favorite_stations']:
            station = self.station_model.find_by_id(station_id)
            if station:
                favorite_stations.append(self.station_model.to_dict(station))
        return favorite_stations
