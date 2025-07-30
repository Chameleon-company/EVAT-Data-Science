import pandas as pd
from fuzzywuzzy import process
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

population_df = pd.read_csv('Suburb_Population.csv')
info_df = pd.read_csv('Info_for_PCZ.csv')
stations_df = pd.read_csv('stations_per_town.csv')

def clean_town_names(df, town_col):
    df[town_col] = df[town_col].astype(str).str.strip().str.lower()
    return df

population_df = clean_town_names(population_df, 'Town')
info_df = clean_town_names(info_df, 'Town')
stations_df = clean_town_names(stations_df, 'Town')


population_df.head()

info_df.head()

stations_df.head()

# Create a mapping from population_df towns to info_df towns
def fuzzy_map_names(source_names, target_names, threshold=90):
    mapped_names = {}
    for name in source_names:
        match, score = process.extractOne(name, target_names)
        if score >= threshold:
            mapped_names[name] = match
        else:
            mapped_names[name] = None
    return mapped_names



# Map population towns to info_df towns
pop_to_info_map = fuzzy_map_names(population_df['Town'].unique(), info_df['Town'].unique())
pop_to_stations_map = fuzzy_map_names(population_df['Town'].unique(), stations_df['Town'].unique())


# Add mapped names as new columns
population_df['Town_info'] = population_df['Town'].map(pop_to_info_map)
population_df['Town_station'] = population_df['Town'].map(pop_to_stations_map)

merged_df = population_df.merge(info_df, left_on='Town_info', right_on='Town', how='left', suffixes=('', '_info'))
merged_df = merged_df.merge(stations_df, left_on='Town_station', right_on='Town', how='left', suffixes=('', '_station'))

# Drop extra town columns 
merged_df.drop(columns=['Town_info', 'Town_station', 'Town_info', 'Town_station'], inplace=True)

merged_df.drop(columns=['Unnamed: 0'], inplace=True)

# Remove dollar signs, commas and convert to numeric
merged_df['Median Weekly Household Income'] = merged_df['Median Weekly Household Income'].replace('[\$,]', '', regex=True).astype(float)
merged_df['All Private Dwellings'] = merged_df['All Private Dwellings'].replace(',', '', regex=True).astype(float)

# Convert vehicle and station counts
merged_df['Average Motor Vehicles per Dwelling'] = merged_df['Average Motor Vehicles per Dwelling'].astype(float)
merged_df['Number of Charging Stations'] = merged_df['Number of Charging Stations'].fillna(0)


# Only keep towns with 1 or more chargers
eligible_df = merged_df[merged_df['Number of Charging Stations'] >= 1].copy()

# Select and copy relevant features
score_df = eligible_df.copy()

# Clean charging stations
score_df['Number of Charging Stations'] = score_df['Number of Charging Stations'].fillna(0)

# Estimate total vehicles
score_df['Estimated Total Vehicles'] = score_df['All Private Dwellings'] * score_df['Average Motor Vehicles per Dwelling']

# Create features for scoring
features = score_df[['Population', 'Median Weekly Household Income', 
                     'Average Motor Vehicles per Dwelling', 'Number of Charging Stations']]



# Normalize (Min-Max Scaling)
scaler = MinMaxScaler()
normalized = scaler.fit_transform(features[['Population', 
                                            'Median Weekly Household Income',
                                            'Average Motor Vehicles per Dwelling',
                                            'Number of Charging Stations']])


# Combine all into final score (you can tweak weights)
score_df['PCZ_Score'] = (
    0.25 * normalized[:, 0] +        # Population
    0.25 * normalized[:, 1] +       # Income
    0.2 * normalized[:, 2] +        # Vehicles/Dwelling
    0.3 * normalized[:, 3]  # More chargers = higher need
)

# Rank towns by score
score_df = score_df.sort_values(by='PCZ_Score', ascending=False)


score_df.head()


# pick top N towns for visual clarity
top_n = 15
df_top = score_df.head(top_n)

# Plotting
plt.figure(figsize=(12, 8))
sns.barplot(data=df_top, x='PCZ_Score', y='Town', palette='crest')
plt.title(f'Top {top_n} Primary Candidate Zones (PCZ) by Score')
plt.xlabel('PCZ Score')
plt.ylabel('Town')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
