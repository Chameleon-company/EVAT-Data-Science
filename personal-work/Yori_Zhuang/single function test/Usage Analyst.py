# Admin: Yulin Zhuang
# Function: visualize charging period
# Last update: 

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

file_path = '../prediction model/EVCS Usage_Sep16_Aug17_Melbourne.csv'
df = pd.read_csv(file_path)

# Convert Start Time and End Time to time format 
# errors='coerce' Filter out different format
df['Start Time'] = pd.to_datetime(df['Start Time'], format='%H:%M', errors='coerce').dt.time
df['End Time'] = pd.to_datetime(df['End Time'], format='%H:%M', errors='coerce').dt.time

# Remove rows with missing Start Time or End Time
df = df.dropna(subset=['Start Time', 'End Time'])

# Convert Start Time and End Time to datetime for duration calculation
df['Start Time'] = df['Start Time'].apply(lambda x: datetime.combine(datetime.today(), x))
df['End Time'] = df.apply(
    lambda row: datetime.combine(datetime.today(), row['End Time'])
    if row['End Time'] > row['Start Time'].time()
    else datetime.combine(datetime.today() + timedelta(days=1), row['End Time']),
    axis=1
)

# Calculate the duration of each charging session (in minutes)
df['Duration (min)'] = (df['End Time'] - df['Start Time']).dt.total_seconds() / 60

# Group data by the hour and count the number of charging sessions per hour
df['Start Hour'] = df['Start Time'].dt.hour
charging_count_per_hour = df.groupby('Start Hour').size()

# Plot the distribution
plt.figure(figsize=(12, 6))
plt.bar(charging_count_per_hour.index, charging_count_per_hour.values, width=0.8, color='purple')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Charging Sessions')
plt.title('Electric Vehicle Charging Sessions by Hour')
plt.xticks(range(24))
plt.grid(axis='y')
plt.show()
