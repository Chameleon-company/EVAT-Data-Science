import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import Choropleth, CircleMarker
import branca.colormap as cm

score_df = pd.read_csv('PCZ.csv')

# Calculate chargers per 1000 people 
score_df['Chargers_per_1000_People'] = (score_df['Number of Charging Stations'] / score_df['Population']) * 1000

# Assuming 5% EV adoption rate
ev_adoption_rate = 0.05
score_df['Estimated_EV_Users'] = score_df['Population'] * ev_adoption_rate

# Normalize new features 
from sklearn.preprocessing import MinMaxScaler

features_to_normalize = ['Chargers_per_1000_People', 'Estimated_EV_Users']
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(score_df[features_to_normalize])

score_df['Norm_Chargers_per_1000'] = normalized_features[:, 0]
score_df['Norm_Estimated_EV_Users'] = normalized_features[:, 1]


# Calculate Final Combined Score 
score_df['Final_Score'] = (
    0.4 * score_df['PCZ_Score'] +          # Original PCZ scoring
    -0.2 * score_df['Norm_Chargers_per_1000'] +  # Negative weight (more chargers = less gap)
    0.4 * score_df['Norm_Estimated_EV_Users']    # Higher future demand = better
)

# Sort by Final Score
score_df = score_df.sort_values(by='Final_Score', ascending=False)

# Final Scores barplot
plt.figure(figsize=(14,7))
sns.barplot(x='Final_Score', y='Town', data=score_df, palette='viridis')
plt.title('Final PCZ Scores after Advanced Validation')
plt.xlabel('Final Score')
plt.ylabel('Town')
plt.tight_layout()
plt.show()

# Chargers per 1000 histogram
plt.figure(figsize=(10,5))
sns.histplot(score_df['Chargers_per_1000_People'], kde=True)
plt.title('Chargers per 1000 People across Top PCZs')
plt.xlabel('Chargers per 1000 People')
plt.ylabel('Frequency')
plt.show()

# EV users estimate histogram
plt.figure(figsize=(10,5))
sns.histplot(score_df['Estimated_EV_Users'], kde=True, color='orange')
plt.title('Estimated EV Users across Top PCZs')
plt.xlabel('Estimated EV Users (5% Adoption)')
plt.ylabel('Frequency')
plt.show()

best_PCZ = score_df.iloc[0]

print("Best Primary Candidate Zone (PCZ) Selected:")
print(f"Town: {best_PCZ['Town']}")
print(f"Final Score: {best_PCZ['Final_Score']:.2f}")
print(f"Estimated EV Users: {int(best_PCZ['Estimated_EV_Users'])}")
print(f"Chargers per 1000 people: {best_PCZ['Chargers_per_1000_People']:.2f}")

lat = [-37.5566, -37.9027, -37.8490, -38.1465, -37.8390, -37.8136, -37.8002, -37.9133, -37.9370, -37.7749, -37.7611, -37.8062, -37.8130, -37.8355, -37.6690]
long = [144.8881, 144.6417, 144.6550, 145.2635, 144.9392, 144.9631, 144.9546, 145.0169, 144.6720, 144.9634, 144.9631, 144.9435, 144.9840, 144.9168, 144.8410]


# Create a base map centered on Melbourne
m = folium.Map(location=[-37.8136, 144.9631], zoom_start=11)

# Create a color map
colormap = cm.linear.YlOrRd_09.scale(score_df['Final_Score'].min(), score_df['Final_Score'].max())
colormap.caption = 'PCZ Score'

count = 0
# Add points
for idx, row in score_df.iterrows():
    folium.CircleMarker(
        location=(lat[count], long[count]),
        radius=7,
        color=colormap(row['Final_Score']),
        fill=True,
        fill_color=colormap(row['Final_Score']),
        fill_opacity=0.7,
        popup=f"Score: {row['Final_Score']:.1f}"
    ).add_to(m)
    count += 1

# Add the colormap legend
colormap.add_to(m)

# Save or display
m.save('melbourne_pcz_map.html')

from IPython.display import IFrame

# Display the map directly in the notebook
IFrame('melbourne_pcz_map.html', width=800, height=600)


