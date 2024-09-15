class UserService:
    def __init__(self, users, charging_stations):
        self.users = users
        self.charging_stations = charging_stations

    def add_user(self, username, email):
        if not username or not email:
            raise ValueError("Username and email are required")

        return self.users.insert(username, email)

    def get_users(self):
        users = self.users.find_all()
        return [self.users.to_dict(user) for user in users]

    def add_favorite_station(self, user_id, station_id):
        user = self.users.find_by_id(user_id)
        if not user:
            raise ValueError("User not found")

        station = self.charging_stations.find_by_id(station_id)
        if not station:
            raise ValueError("Station not found")

        if station_id in user['favorite_stations']:
            raise ValueError("Station is already a favorite")

        return self.users.add_favorite_station(user_id, station_id)

    def remove_favorite_station(self, user_id, station_id):
        user = self.users.find_by_id(user_id)
        if not user:
            raise ValueError("User not found")

        if station_id not in user['favorite_stations']:
            raise ValueError("Station is not a favorite")
        
        station = self.charging_stations.find_by_id(station_id)
        if not station:
            raise ValueError("Station not found")

        return self.users.remove_favorite_station(user_id, station_id)

    def get_favorite_stations(self, user_id):
        user = self.users.find_by_id(user_id)
        if not user:
            raise ValueError("User not found")

        favorite_stations = []
        for station_id in user['favorite_stations']:
            station = self.charging_stations.find_by_id(station_id)
            if station:
                favorite_stations.append(self.charging_stations.to_dict(station))
        return favorite_stations
