
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Page config first
st.set_page_config(page_title="EV Charging Dashboard", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    public_df = pd.read_csv('public_ready_data.csv')
    home_df = pd.read_csv('student_ready_home_data.csv')

    def extract_total_cost(value):
        try:
            if isinstance(value, str):
                value = value.replace('$', '').lower()
                parts = [float(p.strip().replace('/kwh', '').replace('+', '')) for p in value.split('+') if p.strip().replace('/kwh', '').replace('.', '').isdigit()]
                return sum(parts)
            return float(value)
        except:
            return None

    public_df['Cost'] = public_df['UsageCost'].apply(extract_total_cost)
    public_df['Energy_kWh'] = 10
    public_df['Location'] = 'Public'
    public_df['Time_of_Use'] = 'Unknown'

    home_df['Cost'] = home_df['Estimated_Home_Cost_Dollars']
    home_df['Energy_kWh'] = home_df['Home_Energy_kWh']
    home_df['Location'] = 'Home'
    home_df['Time_of_Use'] = home_df['Home_Charge_Time']

    df = pd.concat([
        public_df[['Energy_kWh', 'Cost', 'Location', 'Time_of_Use']],
        home_df[['Energy_kWh', 'Cost', 'Location', 'Time_of_Use']]
    ]).dropna()

    return df

# Load data
df = load_data()

# Title and Description
st.title("⚡ EV Charging Insights Dashboard")
st.markdown("Compare home and public EV charging costs and patterns interactively.")

# 🧠 Help Section
with st.expander("ℹ️ How to use this dashboard"):
    st.markdown("""
        - Use the **sidebar filters** to view insights based on location and time of use.
        - Each chart updates live based on your selected filters.
        - Hover over bars to see exact cost or usage numbers.
        - Use this dashboard to understand user behavior and optimize EV charging strategies.
    """)

# Sidebar Filters
st.sidebar.header("🔍 Filter Data")
location_filter = st.sidebar.multiselect("Charging Location", df['Location'].unique(), default=df['Location'].unique())
time_filter = st.sidebar.multiselect("Time of Use", df['Time_of_Use'].unique(), default=df['Time_of_Use'].unique())

filtered_df = df[(df['Location'].isin(location_filter)) & (df['Time_of_Use'].isin(time_filter))]

# Plot 1: Average Cost by Location
st.subheader("🔌 Average Charging Cost by Location")
st.markdown("This chart compares the average cost of charging at **home vs. public stations**.")
fig1, ax1 = plt.subplots()
sns.barplot(data=filtered_df, x='Location', y='Cost', errorbar=None, ax=ax1)
st.pyplot(fig1)

# Plot 2: Average Cost by Time of Use
st.subheader("🕒 Average Charging Cost by Time of Use")
st.markdown("This chart shows how charging costs vary by **time of day** (e.g., overnight vs. daytime).")
fig2, ax2 = plt.subplots()
sns.barplot(data=filtered_df, x='Time_of_Use', y='Cost', errorbar=None, ax=ax2)
plt.xticks(rotation=30)
st.pyplot(fig2)

# Plot 3: Charging Frequency by Time of Use
st.subheader("📈 Charging Frequency by Time of Use")
st.markdown("This chart shows **how often users charge** during different times of the day.")
fig3, ax3 = plt.subplots()
sns.countplot(data=filtered_df, x='Time_of_Use', order=filtered_df['Time_of_Use'].value_counts().index, ax=ax3)
plt.xticks(rotation=30)
st.pyplot(fig3)
