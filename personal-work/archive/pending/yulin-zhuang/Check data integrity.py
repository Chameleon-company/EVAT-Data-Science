import pandas as pd

# Load the CSV file
df = pd.read_csv('EVCS_Usage_Three_Years.csv')

# Check for negative values in the 'Total kWh' column
negative_total_kwh = df[df['Total kWh'] < 0]
print(f"Found {len(negative_total_kwh)} records with negative 'Total kWh':\n", negative_total_kwh)

# Remove records with negative 'Total kWh'
df_cleaned = df[df['Total kWh'] >= 0]

# Check for missing values
missing_values = df_cleaned.isnull().any().any()

if missing_values:
    print("The dataset contains missing values.")
else:
    print("The dataset has no missing values in any cell.")

# Save the cleaned data to a new CSV file
df_cleaned.to_csv('EVCS_Usage_Three_Years_Cleaned.csv', index=False)

print("Data wrangling successes")
