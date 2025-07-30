import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('weather 2016-09-01 to 2019-05-28.csv', parse_dates=['datetime'])

df['rainy'] = df['preciptype'].fillna('No Rain').apply(lambda x: 'Rain' if x == 'rain' else 'No Rain')

plt.figure(figsize=(12, 6))
plt.plot(df['datetime'], df['humidity'], label='Humidity', color='blue', alpha=0.7)

rainy_days = df[df['rainy'] == 'Rain']
plt.scatter(rainy_days['datetime'], rainy_days['humidity'], color='red', label='Rainy Days', alpha=0.6)

plt.xlabel('Date')
plt.ylabel('Humidity (%)')
plt.title('Humidity Changes Over Time (Highlighted: Rainy Days)')
plt.legend()
plt.grid()
plt.show()