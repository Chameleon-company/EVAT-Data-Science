⚡ EV Charger Demand Forecasting — SIT764 Project
📌 Overview

This project was developed as part of SIT764 — Advanced Data Analytics and Machine Learning at Deakin University.
The goal is to forecast electric vehicle (EV) charger demand in Sydney/NSW, using real-world data, clustering, and machine learning/deep learning models.

The workflow is divided into four main stages:

Data Collection — Gather EV adoption and demand datasets.

Data Cleaning — Preprocess and prepare data for modelling.

Model Building — Train and evaluate forecasting models.

Deployment — Prepare models for real-world use.

📂 Repository Structure
├── Collecting_Data_EV_Charger_Demand.ipynb
├── Cleaning_Data.ipynb
├── Building_Model_For_EV_Charger_Demand_Forecasting.ipynb
├── Deploying_EV_Charger_Demand_Forecasting.ipynb
├── data/
│   ├── merge_data.csv
│   ├── clean_data.csv
└── README.md

⚙️ Dependencies

This project runs on Python 3.10+. Key libraries:

pandas, numpy, matplotlib, seaborn

scikit-learn

statsmodels

prophet

tensorflow / keras

folium (for visualization)

Install requirements with:

pip install -r requirements.txt


(You can generate requirements.txt from your Colab/venv using pip freeze > requirements.txt)

🚀 Workflow
1. Data Collection

Notebook: Collecting_Data_EV_Charger_Demand.ipynb

Mounts Google Drive (Colab).

Collects EV adoption, energy demand, and weather datasets.

Calculates EV adoption percentages.

Output: merge_data.csv

2. Data Cleaning

Notebook: Cleaning_Data.ipynb

Loads merge_data.csv.

Cleans missing values and inconsistencies.

Applies One-Hot Encoding and Scaling.

Computes energy consumption per vehicle type.

Decomposes seasonal trends.

Output: clean_data.csv

3. Model Building

Notebook: Building_Model_For_EV_Charger_Demand_Forecasting.ipynb

Performs clustering (DBSCAN) to group stations.

Forecasting models implemented:

SARIMAX (statistical)

Prophet (seasonal trend model)

LSTM (Deep Learning)

Evaluates models using:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Mean Absolute Percentage Error (MAPE)

Root Mean Squared Error (RMSE)

R² Score

4. Deployment

Notebook: Deploying_EV_Charger_Demand_Forecasting.ipynb

Loads clean_data.csv.

Re-applies clustering for consistency.

Deploys SARIMAX, Prophet, and LSTM in a pipeline.

Prepares models for integration with Streamlit or FastAPI apps.

📊 Results

Classical models (SARIMAX, Prophet) were easier to interpret.

LSTM captured sequential demand patterns better but required more preprocessing.

Outputs can be visualized per geo-cluster of Sydney/NSW.

📝 Notes for Future Students

Save intermediate outputs (merge_data.csv, clean_data.csv) to avoid recomputing.

Always compare at least one classical model (SARIMAX/Prophet) with one deep model (LSTM/GRU).

When scaling, remember: scale input features only, not the target variable.

For deployment, Streamlit or FastAPI are good lightweight options.

Document each step carefully — this makes the project reproducible for others.

👨‍🎓 Authors

Data Science Team (SIT764 Capstone, T2 2025)

Deakin University