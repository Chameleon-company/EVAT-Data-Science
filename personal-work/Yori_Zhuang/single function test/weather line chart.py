import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('weather 2016-09-01 to 2019-05-28.csv', parse_dates=['datetime'])

df = df.sort_values(by='datetime')

max_temp_row = df.loc[df['tempmax'].idxmax()]
min_temp_row = df.loc[df['tempmin'].idxmin()]

plt.figure(figsize=(12, 6))
plt.plot(df['datetime'], df['tempmax'], label='Max Temperature', color='red', alpha=0.7)
plt.plot(df['datetime'], df['tempmin'], label='Min Temperature', color='blue', alpha=0.7)
plt.plot(df['datetime'], df['temp'], label='Average Temperature', color='green', alpha=0.7)

# mark highest and lowest date
plt.scatter(max_temp_row['datetime'], max_temp_row['tempmax'], color='darkred', marker='o', label=f'Highest: {max_temp_row["tempmax"]}°C')
plt.scatter(min_temp_row['datetime'], min_temp_row['tempmin'], color='darkblue', marker='o', label=f'Lowest: {min_temp_row["tempmin"]}°C')
plt.text(max_temp_row['datetime'], max_temp_row['tempmax'], max_temp_row['datetime'].strftime('%Y-%m-%d'), fontsize=10, ha='right', color='darkred')
plt.text(min_temp_row['datetime'], min_temp_row['tempmin'], min_temp_row['datetime'].strftime('%Y-%m-%d'), fontsize=10, ha='right', color='darkblue')

plt.legend()
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Yearly Temperature Trend')
plt.grid()
plt.show()

