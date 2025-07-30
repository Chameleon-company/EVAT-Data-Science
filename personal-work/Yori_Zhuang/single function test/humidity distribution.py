import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('weather 2016-09-01 to 2019-05-28.csv', parse_dates=['datetime'])

df['rainy'] = df['preciptype'].fillna('No Rain').apply(lambda x: 'Rain' if x == 'rain' else 'No Rain')

plt.figure(figsize=(8, 5))
sns.histplot(df[df['rainy'] == 'Rain']['humidity'], color='blue', label='Rain', kde=True, alpha=0.6)
sns.histplot(df[df['rainy'] == 'No Rain']['humidity'], color='orange', label='No Rain', kde=True, alpha=0.6)
plt.xlabel('Humidity (%)')
plt.ylabel('Frequency')
plt.title('Humidity Distribution on Rainy vs Non-Rainy Days')
plt.legend()
plt.grid()
plt.show()