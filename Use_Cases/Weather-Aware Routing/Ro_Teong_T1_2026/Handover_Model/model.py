import math
from config import VEHICLE, DEFAULT_SOC_PCT

def rolling_resistance_adjusted(base_fr, temp_c):
    temp_factor = 1.0 + max(0, (20 - temp_c)) * 0.010
    return base_fr * temp_factor

def compute_bearing(lat1, lng1, lat2, lng2):
    """Calculate compass bearing between two coordinates in degrees."""
    d_lng = math.radians(lng2 - lng1)
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(d_lng) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lng)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360

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

def predict_trip(sections, elevations, weather, vehicle=VEHICLE, ac_on=True):
    total_nominal = 0.0
    for i, step in enumerate(sections):
        elev_delta = 0
        if i + 1 < len(elevations):
            elev_delta = elevations[i+1]["elevation"] - elevations[i]["elevation"]

        # compute accurate bearing per segment (bug 4 fix)
        bearing = compute_bearing(
            step["start_location"]["lat"], step["start_location"]["lng"],
            step["end_location"]["lat"],   step["end_location"]["lng"]
        )

        seg = {
            "distance_m": step["distance"]["value"],
            "duration_s": step["duration"]["value"],
            "elevation_delta_m": elev_delta,
            "bearing_deg": bearing,
        }
        total_nominal += energy_segment_kwh(seg, weather, vehicle)

    # α = 1.40 with AC (Tran et al.), 1.10 without (losses only)
    alpha = 1.40 if ac_on else 1.10
    total_adjusted = total_nominal * alpha
    soc_needed = (total_adjusted / vehicle["battery_capacity_kwh"]) * 100

    return {
        "energy_nominal_kwh": round(total_nominal, 3),       # raw physics, no AC, no traffic
        "energy_with_ac_kwh": round(total_adjusted, 3),      # after AC/losses factor
        "soc_needed_pct": round(soc_needed, 1),
        "soc_with_contingency_pct": round(min(soc_needed * 1.20, 100), 1),
        "ac_on": ac_on,
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
        return 1.0      # light or no traffic
    elif ratio < 1.5:
        return 1.10     # moderate traffic, 10% more energy
    else:
        return 1.20     # heavy traffic, 20% more energy

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