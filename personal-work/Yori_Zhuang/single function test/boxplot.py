import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('weather 2016-09-01 to 2019-05-28.csv', parse_dates=['datetime'])

df['rainy'] = df['preciptype'].fillna('No Rain').apply(lambda x: 'Rain' if x == 'rain' else 'No Rain')

plt.figure(figsize=(8, 5))
sns.boxplot(x='rainy', y='humidity', data=df, palette=['skyblue', 'orange'])
plt.xlabel('Weather Condition')
plt.ylabel('Humidity (%)')
plt.title('Effect of Rain on Humidity')
plt.grid()
plt.show()