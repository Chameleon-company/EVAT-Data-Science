class StationService:
    def __init__(self, station_model):
        self.station_model = station_model

    def add_station(self, station_name, address):
        if not station_name or not address:
            raise ValueError("Station name and address are required")

        existing_station = self.station_model.find_by_name(station_name)
        if existing_station:
            raise ValueError(f"Station '{station_name}' already exists")

        return self.station_model.insert(station_name, address)
    
    def get_stations(self):
        stations = self.station_model.find_all()
        return [self.station_model.to_dict(station) for station in stations]

    def remove_station(self, station_id):
        return self.station_model.delete(station_id)
