
# EV, Traffic & Weather Analysis

This repository contains data and notebooks for exploring the relationship between traffic conditions, EV charging station data, and weather conditipons and a predictive model of the traffic volume.

## Project components
### Datasets

1. **`Weather data.csv`**  
   Contains daily weather observations for a single station (Melbourne Airport) from 2022 to 2023. Important columns:
   - `DATE` – observation date  
   - `PRCP` – daily precipitation (mm)  
   - `TAVG` – average temperature (°C)  
   - `TMAX` – maximum temperature (°C)  
   - `TMIN` – minimum temperature (°C)  

2. **`Traffic data.csv`**  
   Contains road segments with traffic counts, direction, year, and geometry. Important columns:
   - `OBJECTID` – unique ID per segment  
   - `Label` – includes traffic count information (e.g. `450(92)`)  
   - `Direction` – direction of travel (e.g. `both`)  
   - `SHAPE_Length` – segment length in map units  
   - `Year` – year of the traffic count  
   - `geometry` – `LINESTRING` geometry in WKT format  

3. **`EV stations data.csv`**  
   Contains EV charging stations information. Important columns:
   - `OBJECTID` – station ID 
   - `Type` – type of charging station
   - `RatePerHour` – charging rate  
   - `MaxTime` – maximum allowed parking time   

### Notebooks

1. **`Data_Exploration_Analysis.ipynb`**  
   Jupyter notebook used for initial data exploration and analysis of the three datasets (traffic, EV stations, and weather). The work includes:
   - Loading the CSV files  
   - Inspecting three dataset  
   - Visualization

2. **`T3-2025.ipynb`**  
   Main modelling notebook. This notebook:
   - Engineers features from the raw datasets  
   - Builds a combined modelling table  
   - Trains and evaluates a regression model to predict traffic volume  
