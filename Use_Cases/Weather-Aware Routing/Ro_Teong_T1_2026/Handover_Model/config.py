GOOGLE_MAPS_API_KEY = "AIzaSyDnhsnUAhAMPn-Tde44VD0AVoVDpM5kDj0"
# OPEN_CHARGE_MAP_API_KEY = "d4bafc4f-bc9b-4f7f-aaec-f6729f37906a"

VEHICLE = {
    "mass_kg": 1715,
    "Cd": 0.28,
    "frontal_area_m2": 2.32,
    "rolling_resistance": 0.012,
    "wheel_inertia": 0.75,
    "motor_inertia": 0.0384,
    "tyre_radius_m": 0.316,
    "gear_ratio": 8.194,
    "battery_capacity_kwh": 34.5,
    "aux_power_w": 500,
}

DEFAULT_SOC_PCT= 10.0
# this is the default state of charge we assume for the vehicle at
# the start of the trip, if not specified by the user.
# I am not sure if front end supports input, so just a default 
# change to a lower value to see if needs charging  -yes, it does , it worksss