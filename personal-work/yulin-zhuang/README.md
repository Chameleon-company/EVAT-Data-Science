File and folder description

Dataset (raw data folder):
66,000 electric vehicle charging usage data in Perth and Kinross, Scotland
EVCS Usage_Sep16_Aug17_PerthandKinross.csv
EVCS Usage_Sep17_Aug18_PerthandKinross.csv
EVCS Usage_Sep18_Aug19_PerthandKinross.csv

Single function test:
1.charging time analyze:Usage Analyst.py
2.charging kWh analyze:kWh analyze.py

Dataset merging:
Combine the datasets.py
#Merge the three copies of data and redistribute the index values

Data integrity check:
Check data integrity.py
Check whether the dataset contains null values

Merged dataset:
EVCS_Usage_Three_Years.csv

Dataset used by Dash:
EVCS_Usage_Three_Years_Cleaned_Negative_Value.csv

Get coordinates from Google Map API:
Generate site location using Google Map API.py
#Please replace API Key when using

Dash dashboard:
Main dash.py
#I built a Dash dashboard to analyze electric vehicle charging data, including a map for station locations, a histogram for charging times, pie charts, and a heatmap to display energy usage patterns and station frequencies.