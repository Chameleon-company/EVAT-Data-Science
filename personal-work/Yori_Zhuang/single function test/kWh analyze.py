# Admin: Yulin Zhuang
# Function: visualize charging kWh
# Last update:

import pandas as pd
import matplotlib.pyplot as plt

# load csv file
file_path = '../prediction model/EVCS Usage_Sep16_Aug17_Melbourne.csv'
df = pd.read_csv(file_path)

#
df = df.dropna(subset=['Total kWh'])
df = df[df['Total kWh'].apply(lambda x: isinstance(x, (int, float)))]


# calculate the mean of Total kWh
mean_kwh = df['Total kWh'].mean()
sample_count = len(df['Total kWh'])

# histogram using matplotlib
plt.figure(figsize=(12, 6))
plt.hist(df['Total kWh'], bins=20, color='green', edgecolor='black')

# shows annotation at the right corner
annotation_text = f'Mean Total kWh: {mean_kwh:.2f}\nSample Count: {sample_count}'
plt.annotate(annotation_text, xy=(0.95, 0.95), xycoords='axes fraction',
             fontsize=12, color='black', backgroundcolor='white', ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

# setting title and label
plt.xlabel('Total kWh')
plt.ylabel('Frequency')
plt.title('Histogram of Total kWh with Mean Value Annotation')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
