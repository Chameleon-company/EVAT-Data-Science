
from bs4 import BeautifulSoup

import requests

url = 'https://www.citypopulation.de/en/australia/melbourne/'

# Get the page content
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all town rows (each <tr> with class 'rname')
rows = soup.find_all('tr', class_='rname')

# Store data
data_p = []

for row in rows:
    # Get town name
    town = row.find('span', itemprop='name').text.strip()
    
    # Get the latest population (last visible <td class='rpop'>)
    pop_cells = row.find_all('td', class_='rpop')
    latest_pop = pop_cells[-1].text.strip().replace(",", "") if pop_cells else None

    data_p.append({
        'Town': town,
        'Population': int(latest_pop) if latest_pop else None
    })

# Convert to DataFrame
df_pop = pd.DataFrame(data_p)

print(df_pop.head())


df_pop.to_csv("Suburb_Population.csv", index=False)

towns = list(df_pop["Town"])

import re
correct_towns = []
for town in towns:
    # Reformat names like "Altona (North)" → "Altona - North"
    correct_towns.append(re.sub(r"\s*\(([^)]+)\)", r" - \1", town))
