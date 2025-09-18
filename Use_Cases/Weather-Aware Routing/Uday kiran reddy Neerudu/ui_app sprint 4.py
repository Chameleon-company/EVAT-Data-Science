import streamlit as st, requests, folium
from streamlit.components.v1 import html

BASE = "http://127.0.0.1:8000"

st.title("EVAT • Weather-Aware Routing (Sprint 4)")

# fetch station list once
try:
    stations = requests.get(f"{BASE}/stations", timeout=5).json()
    N = len(stations)
    st.caption(f"{N} stations loaded from API")
except Exception as e:
    st.error(f"Could not reach API at {BASE}. Start FastAPI first. Details: {e}")
    st.stop()

col1, col2 = st.columns(2)
start_idx = col1.number_input("Start index", 0, N-1, 0)
goal_idx  = col2.number_input("Goal index", 0, N-1, min(10, N-1))

ev_range   = st.slider("EV range (km)", 20, 120, 70)
k_neigh    = st.slider("K neighbors", 4, 15, 12)
alpha_w    = st.slider("Weather weight", 0.0, 0.5, 0.15, 0.05)
beta_t     = st.slider("Traffic weight", 0.0, 0.5, 0.10, 0.05)
charge_pen = st.slider("Charge penalty (min)", 0, 40, 15)

if st.button("Plan route"):
    payload = dict(
        start_idx=int(start_idx), goal_idx=int(goal_idx),
        ev_range_km=float(ev_range), k_neighbors=int(k_neigh),
        assumed_speed_kmh=60.0,
        alpha_weather=float(alpha_w), beta_traffic=float(beta_t),
        charge_penalty_min=float(charge_pen), mode="FALLBACK"
    )
    try:
        r = requests.post(f"{BASE}/route", json=payload, timeout=10)
        data = r.json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

    st.write("Status:", data.get("status"), "| Hint:", data.get("hint"))
    st.json(data.get("params", {}))

    path = data.get("path", [])
    if not path:
        st.warning("No path. Try increasing EV range or K neighbors.")
    else:
        m = folium.Map(location=[path[0]["lat"], path[0]["lon"]], zoom_start=13)
        coords = []
        for step in path:
            coords.append((step["lat"], step["lon"]))
            folium.Marker(
                (step["lat"], step["lon"]),
                tooltip=f"Step {step['step']} • Node {step['node_id']}"
            ).add_to(m)
        folium.PolyLine(coords, weight=5).add_to(m)
        folium.Marker(coords[-1],
                      tooltip=f"Goal • Cost (min): {data.get('total_cost_min')}").add_to(m)
        html(m._repr_html_(), height=520)
