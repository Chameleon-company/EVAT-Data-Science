import math
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# -------------------------- Page setup --------------------------
st.set_page_config(
    page_title="EVAT ChargeCast — Congestion & Capacity Planner",
    page_icon="🔋",
    layout="wide",
    menu_items={"about": "EVAT Congestion Dashboard • Queueing (M/M/c) with what-if controls"}
)

st.title("🔋 EVAT ChargeCast — Congestion & Capacity Planner")
st.caption("Predict arrivals, recompute queue waits with **Erlang-C**, and test **what-if** scenarios for charger count and service speed.")

# -------------------------- Data loading --------------------------
@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    # Ensure needed columns / safe defaults
    if "lambda_hour" not in df.columns and "pred_arrivals_3h" in df.columns:
        df["lambda_hour"] = np.clip(df["pred_arrivals_3h"], 0, None) / 3.0
    if "c" not in df.columns:
        df["c"] = 1
    if "mu" not in df.columns:
        df["mu"] = np.nan
    # Standardize column presence for charts
    for col in ["arrivals", "pred_arrivals_3h", "expected_wait_mins", "expected_queue_len", "stationId"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

path = "predictions_3h_with_wait_times.csv"
df = load_data(path)

# -------------------------- Sidebar controls --------------------------
st.sidebar.header("Filters")

# 1) Build a mapping: stationId -> friendly name
#    If your DF already has a 'stationName' column, use it; otherwise fall back to a custom dict (edit as needed).
if "stationName" in df.columns:
    id_to_name = (
        df.loc[~df["stationId"].isna(), ["stationId", "stationName"]]
          .drop_duplicates()
          .set_index("stationId")["stationName"]
          .to_dict()
    )
else:
    # 👇 Edit these to your preferred realistic names
    id_to_name = {
        228137: "Melbourne Central – Fast DC Chargers",
        369001: "Docklands Harbour – EV Hub",
        474204: "South Yarra – Shopping Precinct",
        955429: "Brunswick East – Community Hub",
    }

# Fallback for any IDs without a friendly name
def friendly_name(sid):
    return id_to_name.get(sid, f"Station {sid}")

# 2) Build select options as (label, id) so we can show names but keep ids
unique_ids = sorted(df["stationId"].dropna().unique().tolist(), key=lambda x: str(x))
options = [(f"{friendly_name(sid)}", str(sid)) for sid in unique_ids]

# 3) Selectbox shows names; returns the underlying id as string
label_list = [lbl for lbl, _sid in options]
value_list = [sid for _lbl, sid in options]
default_index = 0  # change if you want a different default
selected_label = st.sidebar.selectbox("Station", label_list, index=default_index)
sid = value_list[label_list.index(selected_label)]  # <- underlying stationId (string)

# -------------------------- Station slice & date range --------------------------
d_base = (
    df[df["stationId"].astype(str) == sid]
      .sort_values("timestamp")
      .copy()
)

min_dt, max_dt = d_base["timestamp"].min(), d_base["timestamp"].max()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_dt.date(), max_dt.date()),
    min_value=min_dt.date(),
    max_value=max_dt.date(),
    help="Filter historical data window for the selected station."
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)  # inclusive end
    d_base = d_base[(d_base["timestamp"] >= start_dt) & (d_base["timestamp"] < end_dt)]

# -------------------------- What-if settings --------------------------
st.sidebar.header("What-if settings")
c_multiplier  = st.sidebar.slider("Charger multiplier (c×)", 0.5, 3.0, 1.0, 0.1)
mu_multiplier = st.sidebar.slider("Service speed multiplier (μ×)", 0.5, 2.0, 1.0, 0.1)
roll_window   = st.sidebar.slider("Smoothing (hours)", 0, 6, 2, 1)
wait_target   = st.sidebar.number_input("Target max wait (mins)", min_value=0, value=15, step=5)

# -------------------------- Erlang-C helper --------------------------
def erlang_c_wait_time(lam_hour: float, mu_hour: float, c: int):
    """Return (Wq_hours, Lq, rho). Safe for edge cases."""
    if c <= 0 or mu_hour is None or np.isnan(mu_hour) or mu_hour <= 0:
        return np.nan, np.nan, np.nan
    if lam_hour <= 0:
        return 0.0, 0.0, 0.0
    rho = lam_hour / (c * mu_hour)
    if rho >= 1.0:
        return float("inf"), float("inf"), float(rho)

    a = lam_hour / mu_hour  # offered load (Erlangs)
    # sum_{n=0}^{c-1} a^n / n! computed stably
    sum_terms = 1.0
    term = 1.0
    for n in range(1, c):
        term *= a / n
        sum_terms += term
    term_c = term * (a / c) if c > 0 else 0.0

    P0 = 1.0 / (sum_terms + (term_c / (1.0 - rho)))
    Lq = (P0 * term_c * rho) / ((1.0 - rho) ** 2)
    Wq_hours = Lq / lam_hour
    return float(Wq_hours), float(Lq), float(rho)

# -------------------------- What-if recompute --------------------------
d = d_base.copy()
d["c_adj"]  = np.maximum(1, np.round(d["c"] * c_multiplier).astype(int))
d["mu_adj"] = d["mu"] * mu_multiplier

Wq_mins_adj, rho_adj = [], []
lam_series = d.get("lambda_hour", pd.Series([np.nan] * len(d)))
for lam, mu, cc in zip(lam_series, d["mu_adj"], d["c_adj"]):
    Wq_h, Lq, rho = erlang_c_wait_time(
        float(lam) if pd.notnull(lam) else 0.0,
        float(mu) if pd.notnull(mu) else np.nan,
        int(cc)
    )
    Wq_mins_adj.append(np.nan if np.isinf(Wq_h) else Wq_h * 60.0)
    rho_adj.append(rho)

d["expected_wait_mins_adj"] = Wq_mins_adj
d["rho_adj"] = rho_adj

# Optional smoothing (visual)
if roll_window and roll_window > 0:
    d = d.sort_values("timestamp")
    for col in ["arrivals", "pred_arrivals_3h", "expected_wait_mins", "expected_wait_mins_adj"]:
        if col in d.columns:
            d[col] = d[col].rolling(int(roll_window), min_periods=1).mean()

# -------------------------- Top alert & status --------------------------
missing_service = d["mu"].isna().all()
if missing_service:
    st.warning(
        "This station lacks service-rate (μ) data in the file. "
        "What-if waits will show as **N/A** where μ is missing.",
        icon="⚠️"
    )

if pd.Series(d["rho_adj"]).dropna().ge(1.0).any():
    st.error("System enters **unstable** region (ρ≥1) in the selected window. Increase **c×** or **μ×**.", icon="🔥")

# -------------------------- KPIs --------------------------
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

latest = d.iloc[-1] if len(d) else pd.Series(dtype=float)

# Latest wait
latest_wait = latest.get("expected_wait_mins_adj", np.nan)
if (pd.isna(latest_wait) or latest_wait is None) and "expected_wait_mins" in d.columns:
    latest_wait = latest.get("expected_wait_mins", np.nan)

# Queue length (if provided)
latest_q = latest.get("expected_queue_len", np.nan)
# Utilization
latest_rho = latest.get("rho_adj", np.nan)

# Status chip
def wait_status(w):
    if pd.isna(w):
        return "N/A"
    return "✅ On-target" if w <= wait_target else "⚠️ Above target"

with kpi_col1:
    st.metric("Latest predicted wait (mins)", f"{latest_wait:.1f}" if pd.notnull(latest_wait) else "N/A", help="From Erlang-C with current what-if multipliers")
with kpi_col2:
    st.metric("Latest predicted queue length", f"{latest_q:.2f}" if pd.notnull(latest_q) else "N/A")
with kpi_col3:
    st.metric("Utilization ρ (latest)", f"{latest_rho:.2f}" if pd.notnull(latest_rho) else "N/A", help="ρ = λ / (c·μ)")
with kpi_col4:
    st.metric("Wait status vs target", wait_status(latest_wait), help=f"Target ≤ {wait_target} mins")

# -------------------------- Tabs --------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Time series", "Table", "About"])

# ------ Overview: compact snapshot ------
with tab1:
    left, right = st.columns((1.2, 1), vertical_alignment="top")

    with left:
        st.subheader("Arrivals (actual vs predicted, 3h bins)")
        plot_df = d[["timestamp", "pred_arrivals_3h", "arrivals"]].melt("timestamp", var_name="series", value_name="value")
        chart1 = (
            alt.Chart(plot_df.dropna(subset=["value"]))
            .mark_line(point=False)
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y("value:Q", title="Arrivals (3h)"),
                color=alt.Color("series:N", title="Series", scale=alt.Scale(domain=["arrivals","pred_arrivals_3h"], range=["#4C78A8","#F58518"])),
                tooltip=["timestamp:T","series:N","value:Q"]
            )
            .properties(height=280)
        )
        st.altair_chart(chart1, use_container_width=True)

    with right:
        st.subheader("Expected wait (minutes)")
        wait_col = "expected_wait_mins_adj" if "expected_wait_mins_adj" in d.columns else "expected_wait_mins"
        wait_df = d[["timestamp", wait_col]].rename(columns={wait_col: "wait_mins"})
        rule = alt.Chart(pd.DataFrame({"y": [wait_target]})).mark_rule(strokeDash=[4,4]).encode(y="y:Q")
        chart2 = (
            alt.Chart(wait_df.dropna())
            .mark_line()
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y("wait_mins:Q", title="Expected wait (mins)"),
                tooltip=["timestamp:T","wait_mins:Q"]
            )
            .properties(height=280)
        )
        st.altair_chart(chart2 + rule, use_container_width=True)

    st.info(
        "Tip: If waits exceed target, try **increasing c×** (more chargers) or **increasing μ×** (faster service). "
        "Watch utilization ρ; when ρ ≥ 1 the queue grows without bound.",
        icon="💡"
    )

# ------ Time series: richer exploration ------
with tab2:
    st.subheader("Detailed time series")
    sub_left, sub_right = st.columns(2)

    with sub_left:
        st.markdown("**Arrivals (overlay)**")
        st.altair_chart(chart1.interactive(), use_container_width=True)

    with sub_right:
        st.markdown("**Wait vs target**")
        st.altair_chart((chart2 + rule).interactive(), use_container_width=True)

    # Utilization over time
    util_df = d[["timestamp", "rho_adj"]].rename(columns={"rho_adj": "rho"})
    util_rule = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color="#D62728", strokeDash=[6,3]).encode(y="y:Q")
    util_chart = (
        alt.Chart(util_df.dropna())
        .mark_line()
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("rho:Q", title="Utilization ρ"),
            tooltip=["timestamp:T","rho:Q"]
        )
        .properties(height=240)
    )
    st.markdown("**Utilization (ρ) — stability check**")
    st.altair_chart(util_chart + util_rule, use_container_width=True)

# ------ Table: export & inspect ------
with tab3:
    st.subheader("Data (filtered)")
    show_cols = ["timestamp","stationId","arrivals","pred_arrivals_3h","lambda_hour","c","c_adj","mu","mu_adj","expected_wait_mins","expected_wait_mins_adj","expected_queue_len","rho_adj"]
    present_cols = [c for c in show_cols if c in d.columns]
    st.dataframe(d[present_cols].reset_index(drop=True), use_container_width=True, height=380)

    csv = d[present_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered data (CSV)", data=csv, file_name=f"evat_{sid}_filtered.csv", mime="text/csv")

# ------ About: quick primer ------
with tab4:
    st.markdown("""
### What this dashboard shows
- **Arrivals (3h bins):** actual vs predicted counts.
- **Expected wait:** recomputed using **Erlang-C** under your what-if settings.
- **Utilization (ρ):** load factor; if **ρ ≥ 1** the system is unstable (queues explode).

### Queueing model (M/M/c) recap
- **λ (per hour):** arrival rate (derived from predicted 3h arrivals).
- **μ (per hour):** per-charger service rate (from data; scaled by **μ×**).
- **c:** number of chargers (scaled by **c×**).
- **ρ = λ / (c·μ):** stability indicator.
- **E[Wq] (mins):** expected waiting time in queue from Erlang-C.

### Notes
- If your file lacks μ or c, what-if recomputation is limited. Provide station-level **μ** and **c** for best results.
- Use the **smoothing** slider to de-noise visualization without changing the underlying values.
""")
