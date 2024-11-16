import pandas as pd

# Define the column names
columns = ['_id', 'CP ID', 'Connector', 'Start Date', 'Start Time', 'End Date', 'End Time', 'Total kWh', 'Site', 'Model']

# Load the datasets and skip the first row
df1 = pd.read_csv('EVCS Usage_Sep16_Aug17_PerthandKinross.csv', skiprows=1, header=None)
df2 = pd.read_csv('EVCS Usage_Sep17_Aug18_PerthandKinross.csv', skiprows=1, header=None)
df3 = pd.read_csv('EVCS Usage_Sep18_Aug19_PerthandKinross.csv', skiprows=1, header=None)

# Combine the datasets
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# Redefine column names
combined_df.columns = columns

# Convert 'Start Date' and 'End Date' to datetime format (YYYY-MM-DD)
combined_df['Start Date'] = pd.to_datetime(combined_df['Start Date'], errors='coerce').dt.strftime('%Y-%m-%d')
combined_df['End Date'] = pd.to_datetime(combined_df['End Date'], errors='coerce').dt.strftime('%Y-%m-%d')

# Convert 'Start Time' and 'End Time' to time format (HH:MM:SS)
combined_df['Start Time'] = pd.to_datetime(combined_df['Start Time'], format='%H:%M', errors='coerce').dt.strftime('%H:%M:%S')
combined_df['End Time'] = pd.to_datetime(combined_df['End Time'], format='%H:%M', errors='coerce').dt.strftime('%H:%M:%S')

# Reassign _id sequentially from 1
combined_df['_id'] = range(1, len(combined_df) + 1)

# Save the combined DataFrame to a CSV file
combined_df.to_csv('EVCS_Usage_Three_Years.csv', index=False)

print("Combined DF Shape:", combined_df.shape)
print("Combined DF Head:", combined_df.head())
