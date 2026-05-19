import math
from config import VEHICLE, DEFAULT_SOC_PCT

def rolling_resistance_adjusted(base_fr, temp_c):
    temp_factor = 1.0 + max(0, (20 - temp_c)) * 0.010
    return base_fr * temp_factor

def headwind_component(wind_speed_ms, wind_deg, road_bearing_deg):
    angle_diff = math.radians(road_bearing_deg - wind_deg)
    return wind_speed_ms * math.cos(angle_diff)

def energy_segment_kwh(segment, weather, vehicle=VEHICLE):
    d = segment["distance_m"]
    t = segment["duration_s"]
    if t == 0 or d == 0:
        return 0.0

    v = d / t
    slope_angle = math.atan2(segment["elevation_delta_m"], d)
    rho = 1.225 * (1 - 0.0065 * weather["temp_c"] / 288.15) ** 5.256
    fr  = rolling_resistance_adjusted(vehicle["rolling_resistance"], weather["temp_c"])
    W   = headwind_component(weather["wind_speed_ms"], weather["wind_deg"], segment.get("bearing_deg", 0))

    F_roll = fr * vehicle["mass_kg"] * 9.81 * math.cos(slope_angle)
    F_aero = 0.5 * rho * vehicle["Cd"] * vehicle["frontal_area_m2"] * (v - W) ** 2
    F_hill = vehicle["mass_kg"] * 9.81 * math.sin(slope_angle)

    P_bat = max((F_roll + F_aero + F_hill) * v + vehicle["aux_power_w"], 0)
    return (P_bat * (t / 3600)) / 1000

def predict_trip(sections, elevations, weather, vehicle=VEHICLE):
    total_nominal = 0.0
    for i, step in enumerate(sections):
        elev_delta = 0
        if i + 1 < len(elevations):
            elev_delta = elevations[i+1]["elevation"] - elevations[i]["elevation"]
        seg = {
            "distance_m": step["distance"]["value"],
            "duration_s": step["duration"]["value"],
            "elevation_delta_m": elev_delta,
            "bearing_deg": 0,
        }
        total_nominal += energy_segment_kwh(seg, weather, vehicle)

    total_with_ac = total_nominal * 1.40
    soc_needed = (total_with_ac / vehicle["battery_capacity_kwh"]) * 100

    return {
        "energy_nominal_kwh": round(total_nominal, 3),
        "energy_with_ac_kwh": round(total_with_ac, 3),
        "soc_needed_pct": round(soc_needed, 1),
        "soc_with_contingency_pct": round(min(soc_needed * 1.20, 100), 1),
    }

def traffic_energy_factor(duration_normal_s: float, duration_traffic_s: float) -> float:
    """
    Adjusts energy prediction based on traffic conditions.
    More time in traffic = more stop-start driving = more energy used.
    """
    if duration_normal_s == 0:
        return 1.0
    ratio = duration_traffic_s / duration_normal_s
    if ratio < 1.2:
        return 1.0    # light or no traffic
    elif ratio < 1.5:
        return 1.10   # moderate traffic, 10% more energy
    else:
        return 1.20   # heavy traffic, 20% more energy

def traffic_condition_label(factor: float) -> str:
    """Human readable label for the frontend to display."""
    if factor == 1.0:
        return "light"
    elif factor == 1.10:
        return "moderate"
    else:
        return "heavy"

def needs_charging(soc_needed_pct: float, current_soc_pct: float = DEFAULT_SOC_PCT) -> bool:
    return soc_needed_pct > current_soc_pct