# EV vs ICE Cost Comparison Tool

This directory contains a suite of tools and machine learning scripts designed to compare the operational costs and emissions of Electric Vehicles (EVs) versus Internal Combustion Engine (ICE) vehicles. 

The project includes interactive command-line calculators, synthetic data generators, and predictive forecasting models.

## Files Overview

### 1. Core Applications
- **`app.py`** & **`test.py`**: 
  These are the main interactive command-line tools. They allow a user to select an EV make/model, infer its efficiency, and compare its trip costs against an ICE vehicle. 
  - Features include scraping live Australian electricity and petrol prices (via Finder/Canstar and FuelWatch/FuelCheck), falling back to CSV datasets (`test.charging_stations.csv`), and calculating approximate CO2 emissions saved.
  - **Usage**: Run `python app.py --interactive` to start the guided prompt.

### 2. Data Simulation
- **`dummy_data_generator.py`**: 
  A simulator that generates synthetic interactive transcripts and structured datasets (`dummy_data.csv`, Excel, or text). 
  - Useful for stress-testing the models or application logic by generating randomized realistic (and extreme) trip distances, fluctuating petrol/electricity prices, and various ICE/EV efficiencies.
  - **Usage**: `python dummy_data_generator.py --runs 1000 --csv-out dummy_data.csv`

### 3. Machine Learning & Forecasting
- **`model.py`**: 
  A machine learning pipeline that ingests `dummy_data.csv` to train and evaluate various regression models (Linear, Ridge, Random Forest, Gradient Boosting). 
  - It engineers features (like fuel cost per km, efficiency ratios) and evaluates models to predict expected cost savings. It also includes 10-year scenario forecasting visualizations.
  - **Usage**: `python model.py`

- **`generate_forecast_records.py`**: 
  A production script that uses the best-trained model (saved as a `.joblib` file) to generate large-scale forecast records (e.g., 20k–60k+ rows). 
  - It streams predictions into `forecast_output.csv` based on different economic growth scenarios (low, medium, high) for electricity and petrol over the next decade.
  - **Usage**: `python generate_forecast_records.py`

## Required Datasets (Inputs)
For the tools to work optimally, ensure the following CSVs are present in the directory:
- `test.ev_vehicles.csv`: Contains EV makes, models, variants, battery capacity, and range.
- `test.charging_stations.csv` (Optional): Contains localized charging station pricing.
- `ice_vehicles.csv` (Optional): Used by the API fallback to get specific ICE vehicle fuel efficiencies.
- `dummy_data.csv`: Required for training models in `model.py` and `generate_forecast_records.py` (you can generate this using `dummy_data_generator.py`).

## Setup & Installation

Ensure you have the required Python packages installed:

```bash
pip install pandas numpy scikit-learn matplotlib joblib requests beautifulsoup4 xlsxwriter
```

*Optional APIs:*
To enable NSW FuelCheck data, set the `FUELCHECK_API_KEY` environment variable prior to running the applications.
