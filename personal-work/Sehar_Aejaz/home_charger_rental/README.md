# Home Charger Rental Research

This project explores the feasibility of a **home electric vehicle (EV) charger rental business** in Melbourne, Australia. It includes data collection, cleaning, and analysis of EV charging infrastructure, population, and socio-economic factors across Melbourne suburbs to identify promising locations for such a service.

## Project Overview

The goal of this project is to:
- Identify suburbs with limited public EV charging stations.
- Combine population and vehicle ownership data to gauge potential demand.
- Explore socio-economic metrics (income, dwellings) for targeting areas with optimal customer potential.

---

## Features

### 1. **Charging Station Data**
- Uses Open Charge Map API to fetch EV charger location, type, power, and cost info.
- Stores this in `charger_info.csv`.
- Aggregates the number of chargers per town.

### 2. **Suburb Population Scraping**
- Web scrapes **https://www.citypopulation.de** to get latest population figures per suburb.
- Saves data into `Suburb_Population.csv`.

### 3. **Socioeconomic Indicators**
- Uses Selenium and BeautifulSoup to extract:
  - Median household income
  - Average number of vehicles per dwelling
  - Total private dwellings
- From the **Australian Bureau of Statistics (ABS)** QuickStats site.
- Saved in `Info_for_PCZ.csv`.

### 4. **Data Matching & Cleanup**
- Standardizes and matches suburb names across multiple datasets using fuzzy matching.
- Merges data to support further analytics (e.g. charger availability vs population/income).

### 5. **PCZ Analysis and Validation**
- Calculates a **Priority Charging Zone (PCZ) score** for each town using normalized metrics like population, income, motor vehicles per dwelling, and number of existing chargers.
- Derives additional features:
  - **Chargers per 1000 people**
  - **Estimated EV users** (assuming 5% adoption rate)
- Computes a **final score** by weighing original PCZ score, charger density, and future demand.
- Visualizes results with:
  - Bar plots and histograms of charger density and EV demand
  - **Interactive Folium map** to highlight top PCZs geographically
- Helps identify high-potential locations for launching the rental platform or pilot programs.

---

## Requirements

Install dependencies via:

```bash
pip install pandas requests beautifulsoup4 selenium fuzzywuzzy python-Levenshtein
```

```bash
pip install pandas requests beautifulsoup4 selenium fuzzywuzzy python-Levenshtein
```
Ensure you have [ChromeDriver](https://chromedriver.chromium.org/downloads) installed and available in your system PATH.

---

## API Key

This project uses the [Open Charge Map API](https://openchargemap.org/site/develop/api) for EV charging data. Replace the placeholder API key in the script with your own if needed:

```python
"key": "YOUR_API_KEY"
```

---

## Use Case

The results of this analysis can be used to:
- Launch a **home-based EV charger sharing platform**.
- Identify under-served suburbs for **pilot deployments**.
- Integrate data-driven insights into **urban planning** or **EV infrastructure** investment.

---


