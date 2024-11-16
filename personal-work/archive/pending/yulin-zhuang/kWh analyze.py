# Admin: Yulin Zhuang
# Function: visualize charging kWh
# Last update: 08/07/2024

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load csv file
file_path = 'EVCS Usage_Sep16_Aug17_PerthandKinross.csv'
df = pd.read_csv(file_path)

# calculate the mean of Total kWh
mean_kwh = df['Total kWh'].mean()
sample_count = len(df['Total kWh'])

sns.set(style="whitegrid")
# histogram
plt.figure(figsize=(12, 6))
sns.histplot(df['Total kWh'], bins=20, kde=False, color='green')

# shows annotation at the right corner
annotation_text = f'Mean Total kWh: {mean_kwh:.2f}\nSample Count: {sample_count}'
plt.annotate(annotation_text, xy=(0.95, 0.95), xycoords='axes fraction',
             fontsize=12, color='black', backgroundcolor='white', ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

# setting title and label
plt.xlabel('Total kWh')
plt.ylabel('Frequency')
plt.title('Histogram of Total kWh with Mean Value Annotation')
plt.show()
