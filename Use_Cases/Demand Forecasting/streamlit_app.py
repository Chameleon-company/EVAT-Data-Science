# streamlit_app.py
# How to run locally:
#   streamlit run streamlit_app.py

import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from tensorflow.keras.models import load_model
import joblib

# ===================== PAGE SETUP =====================
st.set_page_config(page_title="EVAT — GRU Forecast by Cluster", page_icon="⚡", layout="wide")
st.title("⚡ EVAT — GRU Forecast per Cluster")
st.caption("Pick a cluster (0–4), tweak external factors, and see the forecast.")

# Column names must match your training pipeline
TIME_COL     = "Date"
CLUSTER_COL  = "geo_cluster"
TARGET_COL   = "estimated_demand_kWh"
EXOG_COLS    = ["public_holiday","school_holiday","is_weekend",
                "Avg_Temp","Avg_Humidity","Avg_Wind"]

EXPECTED_FEATS = 1 + len(EXOG_COLS)  # 7 = 1 target + 6 exogenous features

# ===================== UTILS =====================
@st.cache_data(show_spinner=False)
def load_history(path: str) -> pd.DataFrame:
    """
    Load the historical dataset used to form the model input window.
    Must contain: Date, geo_cluster, estimated_demand_kWh, and all EXOG_COLS.
    """
    df = pd.read_csv(path, parse_dates=[TIME_COL])
    df = df.sort_values([CLUSTER_COL, TIME_COL]).reset_index(drop=True)

    needed = [TIME_COL, CLUSTER_COL, TARGET_COL] + EXOG_COLS
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"`{path}` is missing columns: {miss}")

    # Safety checks for cluster column
    df[CLUSTER_COL] = pd.to_numeric(df[CLUSTER_COL], errors="coerce")
    if df[CLUSTER_COL].isna().any():
        raise ValueError(f"Found NaN in `{CLUSTER_COL}` inside {path}. Please clean the data.")
    if (df[CLUSTER_COL] < 0).any():
        raise ValueError(f"Negative values detected in `{CLUSTER_COL}` (e.g. -1). Please filter them out.")
    return df

def cluster_dir(cid: int) -> str:
    """Support both new (artifacts/clusters/<id>) and legacy (artifacts/cluster_<id>) layouts."""
    p1 = os.path.join("artifacts", "clusters", str(cid))
    p2 = os.path.join("artifacts", f"cluster_{cid}")
    return p1 if os.path.isdir(p1) else p2

def has_artifacts(cid: int) -> bool:
    """Check if model + scaler exist for the given cluster."""
    cdir = cluster_dir(cid)
    return os.path.exists(os.path.join(cdir, "model_gru.keras")) and \
           os.path.exists(os.path.join(cdir, "scaler_all.joblib"))

def artifact_version_key(geo_cluster: int) -> float:
    """
    Use file modification times as a cache-busting key.
    When you overwrite model/scaler/tail, Streamlit will reload them.
    """
    cdir = cluster_dir(geo_cluster)
    mtimes = []
    for p in ["model_gru.keras", "scaler_all.joblib", "tail.npy"]:
        f = os.path.join(cdir, p)
        if os.path.exists(f):
            mtimes.append(os.path.getmtime(f))
    return max(mtimes) if mtimes else 0.0

@st.cache_resource(show_spinner=False)
def load_artifacts_for_cluster(geo_cluster: int, version_key: float):
    """
    Load model + scaler (+ tail if provided) for a cluster.
    `version_key` is only used to invalidate the cache when artifacts change.
    """
    cdir = cluster_dir(geo_cluster)
    mpath = os.path.join(cdir, "model_gru.keras")
    spath = os.path.join(cdir, "scaler_all.joblib")
    tpath = os.path.join(cdir, "tail.npy")
    if not (os.path.exists(mpath) and os.path.exists(spath)):
        raise FileNotFoundError(f"Missing model/scaler under {cdir}")

    model = load_model(mpath)
    scaler = joblib.load(spath)
    tail_scaled = np.load(tpath) if os.path.exists(tpath) else None
    seq_len = model.input_shape[1]
    n_feat  = model.input_shape[2]
    return model, scaler, tail_scaled, seq_len, n_feat

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the features in the exact [TARGET] + EXOG order expected by the model."""
    return df[[TIME_COL, CLUSTER_COL, TARGET_COL] + EXOG_COLS].copy()

def _scale_matrix_like_training(mat: np.ndarray, scaler) -> np.ndarray:
    """
    Scale features exactly like during training.
    Preferred path: 7-column MinMaxScaler (feature-wise).
    Fallback: legacy 1-column scaler (flatten).
    """
    n_in = getattr(scaler, "n_features_in_", None)
    if n_in == mat.shape[1]:
        return scaler.transform(mat)
    # Legacy: the scaler was fit on a flattened single column
    h, w = mat.shape
    flat = mat.reshape(-1, 1)
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(h, w)

def _inverse_vector_like_training(vec: np.ndarray, scaler) -> np.ndarray:
    """
    Inverse-transform the TARGET like during training.
    - 7-column scaler: use scaler.min_[0] and scaler.scale_[0] (MinMax).
    - 1-column scaler: use the legacy inverse_transform on a flattened vector.
    """
    n_in = getattr(scaler, "n_features_in_", None)
    if n_in and n_in > 1:
        return (vec - scaler.min_[0]) / scaler.scale_[0]
    flat = vec.reshape(-1, 1)
    inv = scaler.inverse_transform(flat)
    return inv.reshape(-1)

def make_future_exog_overrides(base_row: pd.Series, horizon: int, overrides: dict) -> pd.DataFrame:
    """
    Build a horizon×EXOG frame by copying the last observed row and
    applying user overrides (holiday flags, weather, etc.).
    """
    rows = []
    for _ in range(horizon):
        r = {c: base_row.get(c, np.nan) for c in EXOG_COLS}
        r.update(overrides)
        rows.append(r)
    return pd.DataFrame(rows)

def scale_future_exog(future_exog_df: pd.DataFrame, scaler, n_feat: int) -> np.ndarray:
    """
    Scale future EXOG (H, 6) the same way as during training.
    If scaler has 7 features, apply MinMax column-wise using the
    parameters of columns 1..6. Otherwise fall back to legacy flatten.
    """
    ex = future_exog_df[EXOG_COLS].to_numpy().astype(np.float32)
    n_in = getattr(scaler, "n_features_in_", None)
    if n_in and n_in >= EXPECTED_FEATS:
        s = scaler.scale_[1:1+len(EXOG_COLS)]
        m = scaler.min_[1:1+len(EXOG_COLS)]
        return ex * s + m  # MinMax forward transform: X_scaled = X * scale_ + min_
    # Legacy fallback
    h, w = ex.shape
    flat = ex.reshape(-1, 1)
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(h, w)

def recursive_forecast(model, scaler, seed_scaled: np.ndarray, exog_future_scaled: np.ndarray, horizon: int) -> np.ndarray:
    """
    One-step recursive forecast (only for legacy one-step models).
    For direct multi-output models (Dense(H)), we don't use this.
    """
    seq_len, n_feat = seed_scaled.shape
    seq = seed_scaled.copy()
    out_scaled = []
    for t in range(horizon):
        x = seq[-seq_len:].reshape(1, seq_len, n_feat)
        yhat_scaled = model.predict(x, verbose=0).ravel()[0]
        next_vec = np.empty((n_feat,), dtype=np.float32)
        next_vec[0] = yhat_scaled
        next_vec[1:] = exog_future_scaled[t]
        seq = np.vstack([seq, next_vec])
        out_scaled.append(yhat_scaled)
    yhat_scaled_arr = np.array(out_scaled, dtype=np.float32)
    return _inverse_vector_like_training(yhat_scaled_arr, scaler)

def infer_freq_from_last_two(ts: pd.Series) -> pd.Timedelta:
    """
    Robust frequency: use the difference between the last two timestamps.
    This avoids gaps if mode()/median() pick a larger interval.
    """
    if len(ts) >= 2:
        return ts.iloc[-1] - ts.iloc[-2]
    return pd.Timedelta(days=1)


# ===================== LOAD DATA =====================
hist_path = "cluster_history.csv"
df_hist = load_history(hist_path)
with st.expander("👀 Inspect cluster_history.csv"):
    st.dataframe(df_hist, use_container_width=True)

if TARGET_COL not in df_hist.columns:
    st.error(f"Column {TARGET_COL} not found in {hist_path}")
    st.stop()

# ===================== SIDEBAR =====================
st.sidebar.subheader("Cluster  ID")
# Allow only 0..4 if present in data *and* artifacts actually exist
allowed = {0, 1, 2, 3, 4}
present = set(df_hist[CLUSTER_COL].unique().tolist()) & allowed
clusters_present = sorted([c for c in present if has_artifacts(c)])
if not clusters_present:
    st.error("No artifacts found for clusters in [0..4].")
    st.stop()

geo_cluster = st.sidebar.selectbox("Cluster (0–4)", clusters_present)

# ---------- External factors (override) ----------
st.sidebar.subheader("External factors (override)")
# Show booleans to the user…
ph_flag = st.sidebar.checkbox("Public holiday", value=False)
sh_flag = st.sidebar.checkbox("School holiday", value=False)
we_flag = st.sidebar.checkbox("Weekend", value=False)

# Convert to ints (0/1) for the model
ph, sh, we = map(int, (ph_flag, sh_flag, we_flag))

tavg = st.sidebar.slider("Avg_Temp (°C)",     -5.0, 45.0, 24.0, 0.5)
havg = st.sidebar.slider("Avg_Humidity (%)",   0.0, 100.0, 60.0, 1.0)
wavg = st.sidebar.slider("Avg_Wind (m/s)",     0.0, 20.0,  3.0, 0.2)

# --- Capacity & alert settings ---
# sensible default: 90th percentile of the chosen cluster's last 60 days
hist_cluster = df_hist[df_hist[CLUSTER_COL] == geo_cluster].sort_values(TIME_COL)
default_cap = int(hist_cluster[TARGET_COL].tail(60).quantile(0.90)) if len(hist_cluster) else 1_000_000

st.sidebar.subheader("Operational limit")
capacity = st.sidebar.number_input(
    "Capacity limit (kWh/day)",
    min_value=0, value=default_cap, step=100_000, format="%i",
    help="Days with forecast above this limit will be flagged."
)
streak_req = st.sidebar.number_input(
    "Capacity alert (days in a row)", min_value=1, max_value=14, value=3, step=1,
    help="Raise an alert if forecast > capacity for this many consecutive days."
)

# ===================== LOAD ARTIFACTS =====================
ver_key = artifact_version_key(int(geo_cluster))  # cache-buster
model, scaler, tail_scaled_opt, SEQ_LEN, N_FEAT = load_artifacts_for_cluster(int(geo_cluster), ver_key)

# Get horizon from model architecture (Dense(H) → H)
out_units = model.output_shape[-1] if isinstance(model.output_shape, tuple) else model.output_shape[0][-1]
is_direct_multi_output = out_units > 1
final_horizon = out_units if is_direct_multi_output else 14

# ===================== SEED =====================
df_feat = build_feature_matrix(df_hist)
seed_raw = (
    df_feat[df_feat[CLUSTER_COL] == geo_cluster]
    .sort_values(TIME_COL)
    .tail(SEQ_LEN)
    .copy()
)
if len(seed_raw) < SEQ_LEN:
    st.error(f"History for cluster {geo_cluster} has less than SEQ_LEN={SEQ_LEN}.")
    st.stop()

# Override EXOG inside the input window using the sidebar controls
seed_raw.loc[:, "public_holiday"] = int(ph)
seed_raw.loc[:, "school_holiday"]  = int(sh)
seed_raw.loc[:, "is_weekend"]      = int(we)
seed_raw.loc[:, "Avg_Temp"]        = float(tavg)
seed_raw.loc[:, "Avg_Humidity"]    = float(havg)
seed_raw.loc[:, "Avg_Wind"]        = float(wavg)

# Scale exactly like training (7-column MinMaxScaler)
seed_mat = seed_raw[[TARGET_COL] + EXOG_COLS].to_numpy().astype(np.float32)
seed_scaled = _scale_matrix_like_training(seed_mat, scaler)

# ===================== FORECAST =====================
if is_direct_multi_output:
    # Direct H-step forecast: one forward pass
    x_in = seed_scaled.reshape(1, SEQ_LEN, EXPECTED_FEATS)
    yhat_scaled = model.predict(x_in, verbose=0).reshape(-1)  # (H,)
    yhat = _inverse_vector_like_training(yhat_scaled, scaler)  # back to kWh
else:
    # Legacy one-step model: recursive roll-out using constant EXOG overrides
    last_row = seed_raw.tail(1).iloc[0]
    overrides = {
        "public_holiday": int(ph),
        "school_holiday": int(sh),
        "is_weekend": int(we),
        "Avg_Temp": float(tavg),
        "Avg_Humidity": float(havg),
        "Avg_Wind": float(wavg),
    }
    future_exog = make_future_exog_overrides(last_row, final_horizon, overrides)
    exog_future_scaled = scale_future_exog(future_exog, scaler, EXPECTED_FEATS)
    yhat = recursive_forecast(model, scaler, seed_scaled, exog_future_scaled, horizon=final_horizon)

# ===================== PLOT =====================
hist_tail = (
    df_hist[df_hist[CLUSTER_COL] == geo_cluster]
    .sort_values(TIME_COL)
    .tail(SEQ_LEN)
    .copy()
)
t0 = hist_tail[TIME_COL].iloc[-1]

# To avoid a visual gap, use the spacing of the last two timestamps
freq = infer_freq_from_last_two(hist_tail[TIME_COL])
future_times = [t0 + (i + 1) * freq for i in range(final_horizon)]

df_plot_hist = pd.DataFrame({
    "timestamp": hist_tail[TIME_COL],
    "value": hist_tail[TARGET_COL],
    "type": "History"
})
df_plot_fcst = pd.DataFrame({
    "timestamp": future_times,
    "value": yhat,
    "type": "Forecast"
})

# Optional: add a join point at t0 using the last actual value so the two lines look continuous
df_plot_fcst = pd.concat([
    pd.DataFrame({"timestamp":[t0],
                  "value":[hist_tail[TARGET_COL].iloc[-1]],
                  "type":["Forecast"]}),
    df_plot_fcst
], ignore_index=True)

df_plot = pd.concat([df_plot_hist, df_plot_fcst], ignore_index=True)

# Figure showing above target
# ---- Exceedance analysis on forecast ----
fcst_only = df_plot_fcst.copy()

# If you added a "join point" at t0, remove it safely (keep only future > t0)
t0 = hist_tail[TIME_COL].iloc[-1]
fcst_only = fcst_only[fcst_only["timestamp"] > t0].copy()

# Compute exceedance flags
fcst_only["exceed"] = fcst_only["value"] > capacity
fcst_only["capacity"] = capacity  # <-- make capacity a DATA FIELD for tooltips

total_exceed = int(fcst_only["exceed"].sum())

# longest consecutive exceedance streak
max_streak, cur = 0, 0
for flag in fcst_only["exceed"].tolist():
    cur = cur + 1 if flag else 0
    max_streak = max(max_streak, cur)

# peak overflow and date
if total_exceed > 0:
    overflows = fcst_only.loc[fcst_only["exceed"], "value"] - capacity
    idx = overflows.idxmax()
    peak_overflow = float(overflows.loc[idx])
    peak_time = fcst_only.loc[idx, "timestamp"]
else:
    peak_overflow, peak_time = 0.0, None

# Base line chart
base_chart = alt.Chart(df_plot).mark_line().encode(
    x=alt.X("timestamp:T", title="Time"),
    y=alt.Y("value:Q", title="Demand (kWh)"),
    color=alt.Color(
        "type:N",
        sort=["History", "Forecast"],
        scale=alt.Scale(
            domain=["History", "Forecast"],
            range=["#1f77b4", "#ff7f0e"]  # blue & orange
        )
    ),
    tooltip=[
        alt.Tooltip("timestamp:T", title="Time"),
        alt.Tooltip("type:N", title="Series"),
        alt.Tooltip("value:Q", title="Demand (kWh)", format=",.0f"),
    ],
).properties(width="container", height=380,
             title=f"Cluster {geo_cluster} — GRU Forecast ({final_horizon} days forward)")

# Horizontal capacity rule (red dashed)
cap_rule = alt.Chart(pd.DataFrame({"y": [capacity]})).mark_rule(
    color="#d62728", strokeDash=[6, 4]
).encode(y="y:Q")

# Red markers on exceedance days (use the 'capacity' FIELD in tooltip)
exceed_points = alt.Chart(fcst_only).transform_filter(
    alt.datum.exceed == True
).mark_point(color="#d62728", size=60, filled=True).encode(
    x="timestamp:T",
    y="value:Q",
    tooltip=[
        alt.Tooltip("timestamp:T", title="Time"),
        alt.Tooltip("value:Q", title="Forecast (kWh)", format=",.0f"),
        alt.Tooltip("capacity:Q", title="Capacity (kWh)", format=",.0f"),
    ],
)

st.altair_chart(base_chart + cap_rule + exceed_points, use_container_width=True)

# ---- Callout ----
if total_exceed > 0:
    msg = f"⚠️ Exceeds capacity on {total_exceed}/{final_horizon} days"
    if max_streak >= int(streak_req):
        msg += f" — longest streak {max_streak} days (≥ {int(streak_req)})."
    if peak_overflow > 0:
        msg += f" Peak overflow {peak_overflow:,.0f} kWh on {peak_time:%b %d}."
    st.error(msg)
else:
    st.success("✅ No capacity exceedances in the forecast window.")




# ===================== EXPORT =====================
with st.expander("Export"):
    st.download_button(
        "Download Forecast CSV",
        data=df_plot_fcst.to_csv(index=False),
        file_name=f"forecast_cluster_{geo_cluster}.csv",
        mime="text/csv"
    )

