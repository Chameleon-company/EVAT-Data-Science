"""
EV vs ICE CO₂ Dashboard + Gradient Boosting Predictions
-------------------------------------------------------
This Streamlit app compares Electric Vehicles (EVs) with Internal
Combustion Engine (ICE) vehicles on CO₂ emissions, trains a
Gradient Boosting model to predict savings, and provides
interactive filters and future-year projections.
"""

# Imports ──────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score


# 1 Page Configuration
# ──────────────────────────────────────────────────────────────
# Set up the overall Streamlit app appearance and theme.
# This defines how the dashboard will look and behave when loaded.
st.set_page_config(
    page_title="EV vs ICE CO₂ Dashboard + GB Predictions",  # Browser tab title
    page_icon="🚗",                                        # Emoji favicon
    layout="wide",                                         # Use full screen width
    initial_sidebar_state="expanded"                       # Sidebar opens by default
)

# Enable Altair dark theme for consistent visual styling across charts
alt.themes.enable("dark")


# 2 Load CSV Files
# ──────────────────────────────────────────────────────────────
# Reads all raw data sources: EV and ICE consumption data,
# plus optional greenhouse emission data for the grid factor.
@st.cache_data  # Cache to improve performance on repeated runs
def load_data():
    ev = pd.read_csv("Data/Pure electric consumption.csv")       # Electric vehicles
    p91 = pd.read_csv("Data/petrol91RON consumption.csv")        # Petrol 91 RON
    p95 = pd.read_csv("Data/petrol95RON consumption.csv")        # Petrol 95 RON
    p98 = pd.read_csv("Data/petrol98RON consumption.csv")        # Petrol 98 RON
    diesel = pd.read_csv("Data/Diesel consumption.csv")          # Diesel vehicles
    # Greenhouse data is optional, used to calculate grid CO₂ factor
    try:
        gh = pd.read_csv("Data/greenhouse.csv")
    except Exception:
        gh = None
    return ev, p91, p95, p98, diesel, gh

# Load and unpack data
ev_raw, p91_raw, p95_raw, p98_raw, diesel_raw, greenhouse_df = load_data()


# 3 Preprocess / Harmonize Columns
# ──────────────────────────────────────────────────────────────
# Standardize column names and calculate derived fields so
# electric and ICE datasets share a common schema.

def prep_ev(df):
    """
    Clean and rename EV dataset columns, ensure numeric types,
    and retain only the required columns for analysis.
    """
    df = df.rename(columns={
        "ModelReleaseYear": "Year",
        "CO2EmissionsCombined": "CO2_g_km",
        "EnergyConsumptionWhkm": "Energy_Wh_km"
    })
    # Convert key columns to numeric, coercing invalid values to NaN
    for c in ["CO2_g_km", "Energy_Wh_km", "Year"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["FuelType"] = "Electric"  # Tag as electric fuel type
    return df[["Make", "Model", "BodyStyle", "Year", "FuelType", "CO2_g_km", "Energy_Wh_km"]]


def prep_ice(df, fuel_name):
    """
    Clean and rename ICE dataset columns, ensure numeric types,
    and compute approximate energy consumption (Wh/km).
    """
    df = df.rename(columns={
        "ModelReleaseYear": "Year",
        "CO2EmissionsCombined": "CO2_g_km",
        "FuelConsumptionCombined": "Fuel_L_100km"
    })
    for c in ["CO2_g_km", "Fuel_L_100km", "Year"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["FuelType"] = fuel_name  # Label the specific ICE fuel type
    # Convert liters/100km to Wh/km (approx 8.9 kWh per liter of fuel)
    if "Fuel_L_100km" in df.columns:
        df["Energy_Wh_km"] = df["Fuel_L_100km"] * 8.9 * 1000 / 100
    else:
        df["Energy_Wh_km"] = np.nan
    return df[["Make", "Model", "BodyStyle", "Year",
               "FuelType", "CO2_g_km", "Energy_Wh_km", "Fuel_L_100km"]].copy()


# Apply preprocessing to all datasets
electric_df = prep_ev(ev_raw)
petrol91 = prep_ice(p91_raw, "Petrol91")
petrol95 = prep_ice(p95_raw, "Petrol95")
petrol98 = prep_ice(p98_raw, "Petrol98")
diesel_df = prep_ice(diesel_raw, "Diesel")

# Combine all ICE data into one dataframe
ice_df = pd.concat([petrol91, petrol95, petrol98, diesel_df], ignore_index=True)


# 4 Compute Grid Emission Factor
# ──────────────────────────────────────────────────────────────
# Calculate kg CO₂ emitted per kWh of electricity generated,
# using greenhouse data for Victoria (VIC). Falls back to 0.9
# kg/kWh if the data is missing or invalid.

def compute_grid_factor(greenhouse_df):
    """
    Returns the electricity grid CO₂ emission intensity
    in kg/kWh based on VIC greenhouse dataset, or a default of 0.9.
    """
    if greenhouse_df is None:
        return 0.9
    try:
        # Filter to VIC state and ensure numeric totals
        vic = greenhouse_df[greenhouse_df['State'].str.upper() == 'VIC'].copy()
        vic['Total emissions (t CO2-e)'] = pd.to_numeric(
            vic['Total emissions (t CO2-e)'].astype(str).str.replace(',', ''), errors='coerce'
        )
        vic['Electricity production (MWh)'] = pd.to_numeric(
            vic['Electricity production (MWh)'].astype(str).str.replace(',', ''), errors='coerce'
        )
        vic = vic.dropna(subset=['Total emissions (t CO2-e)', 'Electricity production (MWh)'])
        if vic.empty:
            return 0.9
        # Compute tonnes CO₂ per MWh, convert to kg/kWh
        emission_intensity_t_per_mwh = vic['Total emissions (t CO2-e)'].sum() / vic['Electricity production (MWh)'].sum()
        return float(emission_intensity_t_per_mwh)
    except Exception:
        return 0.9

# Store grid emission factor for later EV calculations
emission_factor_kg_per_kwh = compute_grid_factor(greenhouse_df)


# 5 Compute EV & ICE Per-Kilometre CO₂ Baselines
# ──────────────────────────────────────────────────────────────
# Add calculated CO₂ emissions for EVs (based on grid factor) and
# ICE vehicles (based on fuel consumption and emission factors).

def add_ev_co2(ev_df, grid_factor):
    """
    Add a column for EV gCO₂/km using grid emission factor.
    Energy_Wh_km * grid_factor (kg/kWh) gives kg/km, multiplied by 1000 = g/km.
    """
    ev_df = ev_df.copy()
    ev_df["Energy_Wh_km"] = pd.to_numeric(ev_df["Energy_Wh_km"], errors="coerce")
    ev_df["EV_gCO2_per_km"] = ev_df["Energy_Wh_km"] * grid_factor
    return ev_df

def add_ice_co2_baseline(ice_df):
    """
    Add a baseline gCO₂/km for ICE vehicles:
      - Petrol ≈ 23.2 gCO₂ per litre per km
      - Diesel ≈ 26.5 gCO₂ per litre per km
    """
    ice_df = ice_df.copy()
    ice_df["ICE_CO2_Baseline_gpkm"] = np.nan
    mask_petrol = ice_df["FuelType"].str.lower().str.contains("petrol", na=False)
    mask_diesel = ice_df["FuelType"].str.lower().str.contains("diesel", na=False)
    if "Fuel_L_100km" in ice_df.columns:
        ice_df.loc[mask_petrol, "ICE_CO2_Baseline_gpkm"] = ice_df.loc[mask_petrol, "Fuel_L_100km"] * 23.2
        ice_df.loc[mask_diesel, "ICE_CO2_Baseline_gpkm"] = ice_df.loc[mask_diesel, "Fuel_L_100km"] * 26.5
    return ice_df

# Apply CO₂ calculations
electric_df = add_ev_co2(electric_df, emission_factor_kg_per_kwh)
ice_df = add_ice_co2_baseline(ice_df)

# Clean rows missing critical info
ev_clean = electric_df.dropna(subset=["Make", "Model", "Year", "EV_gCO2_per_km"])
ice_clean = ice_df.dropna(subset=["Make", "Model", "Year", "ICE_CO2_Baseline_gpkm"])


# 6 Cartesian Join of EV × ICE Pairs
# ──────────────────────────────────────────────────────────────
# Create all combinations of EV and ICE vehicles to compare their
# emissions directly for savings calculations.

ev_pairs = ev_clean[["Make", "Model", "BodyStyle", "Year",
                     "FuelType", "EV_gCO2_per_km", "Energy_Wh_km"]].drop_duplicates(subset=["Make", "Model", "Year"])

ice_pairs = ice_clean[["Make", "Model", "BodyStyle", "Year",
                       "FuelType", "ICE_CO2_Baseline_gpkm", "Energy_Wh_km", "Fuel_L_100km"]].drop_duplicates(subset=["Make", "Model", "Year"])

# To avoid excessive computation, limit size of Cartesian product
MAX_ROWS = 4000
if len(ev_pairs) * len(ice_pairs) > MAX_ROWS:
    ev_pairs = ev_pairs.sample(min(len(ev_pairs), 200), random_state=42)
    ice_pairs = ice_pairs.sample(min(len(ice_pairs), 200), random_state=42)

# Create Cartesian product join
cartesian_df = ev_pairs.assign(key=1).merge(
    ice_pairs.assign(key=1),
    on="key", suffixes=("_EV", "_ICE")
).drop("key", axis=1)


# 7 Feature Engineering & Target Variable
# ──────────────────────────────────────────────────────────────
# Derive additional predictors and the target for the model:
#  - YearDiff: difference in release years
#  - CO2_saving_gpkm: baseline ICE minus EV emissions
#  - EnergyDiff_Wh_km: difference in energy consumption

cartesian_df["YearDiff"] = cartesian_df["Year_EV"].astype(int) - cartesian_df["Year_ICE"].astype(int)
cartesian_df["CO2_saving_gpkm"] = cartesian_df["ICE_CO2_Baseline_gpkm"] - cartesian_df["EV_gCO2_per_km"]
cartesian_df["EnergyDiff_Wh_km"] = cartesian_df["Energy_Wh_km_ICE"].fillna(0) - cartesian_df["Energy_Wh_km_EV"].fillna(0)
cartesian_df = cartesian_df.dropna(subset=["CO2_saving_gpkm"])


# 8 Build & Train Gradient Boosting Model
# ──────────────────────────────────────────────────────────────
# Predict CO₂ savings using categorical and numeric features.
# Model is cross-validated to assess performance.

categorical_cols = ["Make_EV", "Make_ICE", "BodyStyle_EV", "BodyStyle_ICE", "FuelType_ICE"]
numeric_cols = ["YearDiff", "ICE_CO2_Baseline_gpkm", "EnergyDiff_Wh_km"]

# Ensure numeric columns exist
for c in numeric_cols:
    if c not in cartesian_df.columns:
        cartesian_df[c] = 0

X = cartesian_df[categorical_cols + numeric_cols].fillna(0)
y = cartesian_df["CO2_saving_gpkm"]

# One-hot encode categorical variables, pass through numeric
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)],
    remainder="passthrough"
)

gb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(random_state=42, n_estimators=200))
])

@st.cache_data
def train_and_evaluate(X, y):
    """
    Perform 5-fold cross-validation and fit the Gradient Boosting model.
    Returns trained model and R²/MAE metrics.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = cross_val_score(gb_model, X, y, cv=kf, scoring="r2")
    mae_scores = -cross_val_score(gb_model, X, y, cv=kf, scoring="neg_mean_absolute_error")
    gb_model.fit(X, y)
    return gb_model, r2_scores, mae_scores

with st.spinner("Training Gradient Boosting model (this may take a moment)..."):
    try:
        trained_model, r2_scores, mae_scores = train_and_evaluate(X, y)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        trained_model = None
        r2_scores = np.array([np.nan])
        mae_scores = np.array([np.nan])

# Store predictions (fallback to actual savings if model fails)
if trained_model is not None:
    cartesian_df["Predicted_CO2_Saving_gpkm"] = trained_model.predict(X)
else:
    cartesian_df["Predicted_CO2_Saving_gpkm"] = cartesian_df["CO2_saving_gpkm"]


# 9 Sidebar Filters & Projection Settings
# ──────────────────────────────────────────────────────────────
# Sidebar UI for user-driven filtering and future-year projections.

st.sidebar.header("Filters & Projection Settings")

fuel_filter = st.sidebar.multiselect(
    "Select Fuel Type(s)",
    options=sorted(list(cartesian_df["FuelType_EV"].unique()) +
                   list(cartesian_df["FuelType_ICE"].unique())),
    default=["Electric", "Petrol91", "Petrol95", "Petrol98", "Diesel"]
)

body_filter = st.sidebar.multiselect(
    "Body style (EV)",
    options=sorted(cartesian_df["BodyStyle_EV"].dropna().unique())
)

year_min = int(cartesian_df[["Year_EV", "Year_ICE"]].min().min())
year_max = int(cartesian_df[["Year_EV", "Year_ICE"]].max().max())
year_range = st.sidebar.slider("Select Model Year Range", year_min, year_max, (2015, 2025))

future_year = st.sidebar.slider("Future target year for projection", 2025, 2050, 2030)
ev_improve_pct = st.sidebar.slider("Annual EV improvement (fraction)", 0.00, 0.10, 0.02, step=0.005)
ice_improve_pct = st.sidebar.slider("Annual ICE improvement (fraction)", 0.00, 0.05, 0.01, step=0.005)

# Apply selected filters to the data
mask = cartesian_df["FuelType_EV"].isin(fuel_filter) | cartesian_df["FuelType_ICE"].isin(fuel_filter)
mask &= cartesian_df["BodyStyle_EV"].isin(body_filter) if body_filter else True
mask &= cartesian_df["Year_EV"].between(year_range[0], year_range[1]) | \
        cartesian_df["Year_ICE"].between(year_range[0], year_range[1])
filtered_pairs = cartesian_df[mask].copy()

# Projection calculations based on user’s future year and improvement rates
reference_year = int(max(cartesian_df["Year_EV"].max(), cartesian_df["Year_ICE"].max()))
years_forward = max(future_year - reference_year, 0)

filtered_pairs["ICE_current_gpkm"] = filtered_pairs["ICE_CO2_Baseline_gpkm"]
filtered_pairs["EV_current_gpkm"] = filtered_pairs["EV_gCO2_per_km"]
filtered_pairs["ICE_future_gpkm"] = filtered_pairs["ICE_current_gpkm"] * ((1 - ice_improve_pct) ** years_forward)
filtered_pairs["EV_future_gpkm"] = filtered_pairs["EV_current_gpkm"] * ((1 - ev_improve_pct) ** years_forward)

filtered_pairs["Delta_future_minus_current"] = (
    (filtered_pairs["ICE_future_gpkm"] - filtered_pairs["EV_future_gpkm"]) -
    (filtered_pairs["ICE_current_gpkm"] - filtered_pairs["EV_current_gpkm"])
)

filtered_pairs["Predicted_Saving_Current_gpkm"] = filtered_pairs["Predicted_CO2_Saving_gpkm"]
filtered_pairs["Predicted_Saving_Future_gpkm"] = (
    filtered_pairs["Predicted_Saving_Current_gpkm"] + filtered_pairs["Delta_future_minus_current"]
)


# 10 Dashboard Visuals
# ──────────────────────────────────────────────────────────────
# This section creates all the interactive charts and visual summaries
# displayed on the Streamlit dashboard. These visuals allow users to
# explore current vs. projected CO₂ savings, historical trends, and
# performance by EV make and ICE fuel type.

# Set the main page title
st.title("🚗 EV vs ICE — CO₂ Savings (Gradient Boosting Predictions)")

# Display overall model performance (R² and MAE from cross-validation)
st.markdown(
    f"**Model cross-validation:** "
    f"Mean R² = `{np.nanmean(r2_scores):.3f}`, "
    f"Mean MAE = `{np.nanmean(mae_scores):.2f} g/km`"
)

# 10.1 Average Predicted CO₂ Savings by ICE Fuel Type (Future)
# ──────────────────────────────────────────────────────────────
# Shows mean projected savings (g/km) by ICE fuel category for the
# selected filters and projection year.
st.subheader("📊 Avg Predicted CO₂ Savings (Future) by ICE Fuel Type (Filtered)")
if not filtered_pairs.empty:
    agg_fuel = (
        filtered_pairs.groupby("FuelType_ICE")["Predicted_Saving_Future_gpkm"]
        .mean()
        .reset_index()
        .sort_values("Predicted_Saving_Future_gpkm", ascending=False)
    )
    fig_fuel = px.bar(
        agg_fuel,
        x="FuelType_ICE",
        y="Predicted_Saving_Future_gpkm",
        text_auto=".1f",
        color="Predicted_Saving_Future_gpkm",
        color_continuous_scale="Viridis",
        title=f"Average Predicted CO₂ Savings in {future_year} by ICE Fuel Type"
    )
    st.plotly_chart(fig_fuel, use_container_width=True)
else:
    st.info("No data available for the selected filters.")

# 10.2 Historical CO₂ Emissions Trend
# ──────────────────────────────────────────────────────────────
# Compares average historical g/km emissions of EVs vs. ICE vehicles
# over time to visualize the long-term downward trend.
st.subheader("📈 Historical CO₂ Emissions Trend (Average)")
hist_ev = (
    ev_clean.groupby("Year")["EV_gCO2_per_km"]
    .mean()
    .reset_index()
    .rename(columns={"EV_gCO2_per_km": "CO2_g_km"})
)
hist_ev["FuelType"] = "Electric"

hist_ice = (
    ice_clean.groupby("Year")["ICE_CO2_Baseline_gpkm"]
    .mean()
    .reset_index()
    .rename(columns={"ICE_CO2_Baseline_gpkm": "CO2_g_km"})
)
hist_ice["FuelType"] = "ICE"

hist_all = pd.concat([hist_ev, hist_ice], ignore_index=True)
fig_hist = px.line(
    hist_all,
    x="Year",
    y="CO2_g_km",
    color="FuelType",
    markers=True,
    title="Average Historical CO₂ Emissions (g/km)"
)
st.plotly_chart(fig_hist, use_container_width=True)

# 10.3 Top EV Makes by Predicted Future CO₂ Savings
# ──────────────────────────────────────────────────────────────
# Highlights the top 10 EV manufacturers expected to deliver the greatest
# average g/km savings by the chosen projection year.
st.subheader(f"🏆 Top EV Makes by Predicted CO₂ Savings in {future_year}")
if not filtered_pairs.empty:
    top_ev = (
        filtered_pairs.groupby("Make_EV")["Predicted_Saving_Future_gpkm"]
        .mean()
        .reset_index()
        .sort_values("Predicted_Saving_Future_gpkm", ascending=False)
        .head(10)
    )
    fig_top_ev = px.bar(
        top_ev,
        x="Make_EV",
        y="Predicted_Saving_Future_gpkm",
        text_auto=".1f",
        color="Predicted_Saving_Future_gpkm",
        color_continuous_scale="Viridis",
        title=f"Top 10 EV Makes by Predicted CO₂ Savings in {future_year}"
    )
    st.plotly_chart(fig_top_ev, use_container_width=True)
else:
    st.info("No EVs available for the selected filters.")

# 10.4 Scatter: Predicted vs Current CO₂ Savings
# ──────────────────────────────────────────────────────────────
# Compares current predicted savings to projected future savings to
# visualize potential improvements for each EV–ICE pairing.
st.subheader("🔹 Predicted vs Current CO₂ Savings (Filtered)")
if not filtered_pairs.empty:
    fig_scatter = px.scatter(
        filtered_pairs,
        x="Predicted_Saving_Current_gpkm",
        y="Predicted_Saving_Future_gpkm",
        color="FuelType_ICE",
        hover_data=["Make_EV", "Model_EV", "Make_ICE", "Model_ICE"],
        title=f"Predicted vs Future CO₂ Savings in {future_year}",
        labels={
            "Predicted_Saving_Current_gpkm": "Current Predicted Saving (g/km)",
            "Predicted_Saving_Future_gpkm": "Future Predicted Saving (g/km)"
        }
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("No data available for scatter plot with current filters.")