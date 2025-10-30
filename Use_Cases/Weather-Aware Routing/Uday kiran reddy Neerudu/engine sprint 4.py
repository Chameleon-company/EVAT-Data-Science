import pandas as pd
import numpy as np
from collections import defaultdict
import heapq
from .utils import haversine_km, minmax_norm
from .adapters import get_weather_score, get_traffic_proxy

DEFAULTS = {
    "EV_RANGE_KM": 35.0, "K_NEIGHBORS": 6, "ASSUMED_SPEED_KMH": 60.0,
    "ALPHA_WEATHER": 0.15, "BETA_TRAFFIC": 0.10, "CHARGE_TIME_MIN": 15.0,
    "MODE": "FALLBACK",
}

REQ_COLS = {
    "id": ["InfrastructureID","SiteID","StationID","ID","id"],
    "lat": ["Latitude","lat","LAT","Lat"],
    "lon": ["Longitude","lon","LON","Long","Lng"]
}

def pick(df, names):
    for n in names:
        if n in df.columns: return n
    return None

def load_dataset(path):
    df = pd.read_csv(path)

    # Weather score (if missing)
    if "Weather_Sensitivity_Score" not in df.columns:
        for c in ("TMAX","TMIN","PRCP"):
            if c not in df.columns: df[c] = 0.0
        df["Temp_Range"] = df.get("Temp_Range", df["TMAX"] - df["TMIN"])
        df["High_Temp_Flag"] = (pd.to_numeric(df["TMAX"], errors="coerce") > 35).astype(int)
        df["Low_Temp_Flag"]  = (pd.to_numeric(df["TMIN"], errors="coerce") < 5).astype(int)
        df["No_Precip_Flag"] = (pd.to_numeric(df["PRCP"], errors="coerce") == 0).astype(int)

        def tr_score(x):
            if pd.isna(x): return 0.5
            if x > 20: return 1.0
            if x < 10: return 0.0
            return 0.5

        df["Temp_Range_Score"] = pd.to_numeric(df["Temp_Range"], errors="coerce").apply(tr_score)
        df["Weather_Sensitivity_Score"] = (
            2.0*df["High_Temp_Flag"] + 1.5*df["Low_Temp_Flag"] + (1 - df["No_Precip_Flag"]) + df["Temp_Range_Score"]
        )
    df["Weather_Score_Norm"] = np.clip(pd.to_numeric(df["Weather_Sensitivity_Score"], errors="coerce")/5.0,0,1)

    # Traffic proxy (robust)
    if "Traffic_Proxy" not in df.columns:
        if "Congestion_Factor" in df.columns:
            df["Traffic_Proxy"] = pd.to_numeric(df["Congestion_Factor"], errors="coerce")
        elif "Traffic_Volume" in df.columns and "Traffic_Length" in df.columns:
            vol = pd.to_numeric(df["Traffic_Volume"], errors="coerce")
            seg = pd.to_numeric(df["Traffic_Length"], errors="coerce").replace(0, np.nan)
            df["Traffic_Proxy"] = vol / seg
        else:
            df["Traffic_Proxy"] = 0.0
    df["Traffic_Proxy_Norm"] = minmax_norm(pd.to_numeric(df["Traffic_Proxy"], errors="coerce"))
    return df

def build_nodes(df):
    id_col  = pick(df, REQ_COLS["id"])
    lat_col = pick(df, REQ_COLS["lat"])
    lon_col = pick(df, REQ_COLS["lon"])
    if not lat_col or not lon_col:
        raise ValueError("Latitude/Longitude not found in dataset.")

    # If there is no usable ID column, create a synthetic one
    if id_col is None:
        df = df.copy()
        df["__synthetic_id__"] = df.index.astype(str)
        id_col = "__synthetic_id__"

    cols = [id_col, lat_col, lon_col, "Weather_Score_Norm", "Traffic_Proxy_Norm"]
    nodes = df[cols].dropna().copy()
    nodes = nodes.drop_duplicates(subset=[id_col]).reset_index(drop=True)
    nodes.rename(columns={id_col:"node_id", lat_col:"lat", lon_col:"lon"}, inplace=True)
    return nodes

def build_edges(nodes, K_NEIGHBORS, EV_RANGE_KM):
    coords = nodes[["lat","lon"]].to_numpy()
    edges = []
    for i, (la, lo) in enumerate(coords):
        d = haversine_km(la, lo, coords[:,0], coords[:,1])
        idx = np.argsort(d)[1:K_NEIGHBORS+1]
        for j in idx:
            dist_km = float(d[j])
            if dist_km <= EV_RANGE_KM:
                edges.append((i, j, dist_km))
    return edges

def base_time_minutes(dist_km, speed_kmh):
    return 60.0 * dist_km / max(speed_kmh, 1e-6)

def build_costs(nodes, edges, speed_kmh, alpha, beta, mode="FALLBACK"):
    costs = []
    for (u, v, dist_km) in edges:
        travel_min = base_time_minutes(dist_km, speed_kmh)
        w = get_weather_score(nodes.loc[v,"lat"], nodes.loc[v,"lon"], mode=mode,
                              default=float(nodes.loc[v,"Weather_Score_Norm"]))
        t = get_traffic_proxy(nodes.loc[v,"lat"], nodes.loc[v,"lon"], mode=mode,
                              default=float(nodes.loc[v,"Traffic_Proxy_Norm"]))
        cost_min = travel_min * (1 + alpha*w + beta*t)
        costs.append((u, v, dist_km, travel_min, cost_min))
    return costs

def dijkstra_with_stop_penalty(nodes, costs, start_idx, goal_idx, charge_penalty_min):
    from collections import defaultdict
    import heapq

    N = len(nodes)
    if not (0 <= start_idx < N and 0 <= goal_idx < N):
        return None, float("inf")

    adj = defaultdict(list)
    for (u, v, dist_km, travel_min, cost_min) in costs:
        adj[u].append((v, cost_min, dist_km, travel_min))

    dist = [float('inf')]*N
    prev = [-1]*N
    dist[start_idx] = 0.0
    pq = [(0.0, start_idx)]
    visited = set()

    while pq:
        cur_cost, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == goal_idx:
            break
        for v, edge_cost, _dkm, _tmin in adj[u]:
            add = charge_penalty_min if v != goal_idx else 0.0
            new_cost = cur_cost + edge_cost + add
            if new_cost < dist[v]:
                dist[v] = new_cost
                prev[v] = u
                heapq.heappush(pq, (new_cost, v))

    if dist[goal_idx] == float('inf'):
        return None, float('inf')

    path = []
    x = goal_idx
    while x != -1:
        path.append(x)
        x = prev[x]
    return path[::-1], dist[goal_idx]

def summarize_route(nodes, path):
    if not path: return []
    out = []
    for i, n in enumerate(path):
        out.append({
            "step": i,
            "node_index": int(n),
            "node_id": nodes.loc[n,"node_id"],
            "lat": float(nodes.loc[n,"lat"]),
            "lon": float(nodes.loc[n,"lon"]),
        })
    return out

def _safe_total_cost(val):
    try:
        if val is None or val == float("inf") or val != val:
            return None
        return float(val)
    except Exception:
        return None

def plan_route(
    data_path="data/combined_data.csv",
    start_idx=0, goal_idx=10,
    EV_RANGE_KM=DEFAULTS["EV_RANGE_KM"], K_NEIGHBORS=DEFAULTS["K_NEIGHBORS"],
    ASSUMED_SPEED_KMH=DEFAULTS["ASSUMED_SPEED_KMH"],
    ALPHA_WEATHER=DEFAULTS["ALPHA_WEATHER"], BETA_TRAFFIC=DEFAULTS["BETA_TRAFFIC"],
    CHARGE_TIME_MIN=DEFAULTS["CHARGE_TIME_MIN"], MODE=DEFAULTS["MODE"]
):
    df = load_dataset(data_path)
    nodes = build_nodes(df)
    edges = build_edges(nodes, K_NEIGHBORS, EV_RANGE_KM)
    costs = build_costs(nodes, edges, ASSUMED_SPEED_KMH, ALPHA_WEATHER, BETA_TRAFFIC, mode=MODE)
    path, total_cost = dijkstra_with_stop_penalty(nodes, costs, start_idx, goal_idx, CHARGE_TIME_MIN)

    status = "ok"
    hint = None
    if total_cost == float("inf") or not path:
        status = "unreachable"
        hint = "Increase ev_range_km or k_neighbors; or choose closer indices."

    return {
        "status": status,
        "hint": hint,
        "params": dict(EV_RANGE_KM=EV_RANGE_KM, K_NEIGHBORS=K_NEIGHBORS,
                       ASSUMED_SPEED_KMH=ASSUMED_SPEED_KMH,
                       ALPHA_WEATHER=ALPHA_WEATHER, BETA_TRAFFIC=BETA_TRAFFIC,
                       CHARGE_TIME_MIN=CHARGE_TIME_MIN, MODE=MODE),
        "nodes_count": len(nodes),
        "edges_count": len(edges),
        "path": summarize_route(nodes, path) if path else [],
        "total_cost_min": _safe_total_cost(total_cost),
    }
