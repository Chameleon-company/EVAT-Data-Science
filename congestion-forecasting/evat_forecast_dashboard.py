
import math
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

st.set_page_config(page_title="EVAT — Forecasting & What‑If", page_icon="⚡", layout="wide")

st.title("⚡ EVAT — Forecasting & What‑If (Future Dates)")

@st.cache_data
def load_data():
    df = pd.read_csv("forecast_results.csv", parse_dates=["bin_time"])
    return df

df = load_data()
stations = sorted(df["station_id"].unique().tolist())
col1, col2, col3, col4 = st.columns(4)
station = col1.selectbox("Station", stations, index=0)
max_h = df[df["station_id"]==station].shape[0]
horizon = col2.slider("Horizon (future steps)", min_value=8, max_value=max_h, value=min(56, max_h), step=4)
c = col3.number_input("Servers (c)", min_value=1, max_value=50, value=4)
mu = col4.number_input("Service rate (μ per server per hour)", min_value=0.1, max_value=20.0, value=2.0, step=0.1, format="%.1f")

sub = df[df["station_id"]==station].sort_values("bin_time").head(horizon).copy()

def erlang_c_probability_wait(lmbda, mu, c):
    if lmbda <= 0 or mu <= 0 or c <= 0:
        return 0.0
    rho = lmbda / (c * mu)
    if rho >= 1.0:
        return 1.0
    sum_terms = sum([(lmbda/mu)**n / math.factorial(n) for n in range(c)])
    last_term = ((lmbda/mu)**c) / (math.factorial(c) * (1 - rho))
    P0 = 1.0 / (sum_terms + last_term)
    Pw = last_term * P0
    return min(max(Pw, 0.0), 1.0)

def mmc_metrics(lmbda, mu, c):
    if lmbda <= 0 or mu <= 0 or c <= 0:
        return 0.0, 0.0, 0.0, 0.0
    rho = lmbda / (c * mu)
    if rho >= 1.0:
        return rho, 1.0, float("inf"), float("inf")
    Pw = erlang_c_probability_wait(lmbda, mu, c)
    Lq = Pw * rho / (1 - rho)
    Wq = Lq / lmbda if lmbda > 0 else 0.0
    return rho, Pw, Lq, Wq

vals = np.array([mmc_metrics(x, mu, c) for x in sub["lambda_forecast"].values])
sub["rho"] = vals[:,0]
sub["p_wait"] = vals[:,1]
sub["Lq"] = vals[:,2]
sub["Wq_minutes"] = vals[:,3] * 60.0

st.markdown("### λ Forecast with 80% Interval")
line = alt.Chart(sub).mark_line().encode(x="bin_time:T", y="lambda_forecast:Q")
band = alt.Chart(sub).mark_area(opacity=0.2).encode(x="bin_time:T", y="lambda_lower:Q", y2="lambda_upper:Q")
st.altair_chart(band + line, use_container_width=True)

colA, colB = st.columns(2)
with colA:
    st.markdown("### Probability of Waiting")
    st.altair_chart(alt.Chart(sub).mark_line().encode(x="bin_time:T", y="p_wait:Q"), use_container_width=True)
with colB:
    st.markdown("### Expected Wait (minutes)")
    st.altair_chart(alt.Chart(sub.assign(Wq_capped=sub["Wq_minutes"].clip(upper=120))).mark_line().encode(x="bin_time:T", y="Wq_capped:Q"), use_container_width=True)

st.caption("Tune **c** and **μ** to demonstrate operational strategies in your pitch.")
