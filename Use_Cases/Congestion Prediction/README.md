# ⚡ EVAT — Forecasting Congestion in Electric Vehicle Charging Stations

## 📌 Overview
This project is part of the **Electric Vehicle Adoption Tools (EVAT)** initiative at Deakin University.  
It addresses the challenge of **forecasting congestion at EV charging stations** by integrating **Queueing Theory** with **time-series forecasting models**.

As EV adoption accelerates, charging infrastructure faces rising demand, which can lead to long waiting times and reduced user satisfaction. This project provides a **reproducible pipeline, predictive models, and interactive dashboards** to support policymakers, operators, and urban planners.

---

## 🎯 Key Contributions
- **Preprocessing Pipeline**: Aggregated raw charging data into 3-hour bins and engineered temporal + station-based features.  
- **Baseline Models**: Naïve, Seasonal Naïve, and Moving Average for benchmarking.  
- **Advanced Models**: **SARIMAX** and **Poisson-LSTM** to capture seasonality, stochastic arrivals, and non-linear demand.  
- **Queueing Theory Integration**: Applied **M/M/c Erlang-C** to translate demand forecasts into **waiting times (Wq)** and **utilisation (ρ)**.  
- **Interactive Dashboards**: Developed in **Streamlit**, with scenario testing for charger/service multipliers.  

---

## 📊 Results
| Model            | MAE   | RMSE  | MAPE   |
|------------------|-------|-------|--------|
| Naïve            | 12.3  | 15.6  | 22.1%  |
| Seasonal Naïve   | 10.7  | 13.9  | 18.5%  |
| Moving Average   | 11.5  | 14.8  | 19.7%  |
| SARIMAX          | 7.8   | 10.4  | 14.2%  |
| **LSTM (Poisson)** | **6.2** | **8.9** | **11.5%** |

- **LSTM delivered the highest accuracy**, outperforming both baseline and SARIMAX models.  
- Erlang-C analysis enabled **operational insights** such as waiting times and utilisation.  
- Dashboards provide **explorable forecasts** and support real-time decision making.  

---

## 🏗️ Project Structure
```
CONGESTION PREDICTION/
├── .ipynb_checkpoints/            # Jupyter autosave checkpoints
├── artifacts/                     # Saved model artifacts
├── artifacts_premium/             # Extended/alternative experiment outputs
├── reports/                       
│   └── EVAT_Congestion_Prediction_Report.pdf  # Full technical report
│
├── Data preprocessing.ipynb       # Data cleaning & feature engineering
├── EVAT_Congestion_with_baselines_models.ipynb   # Naïve, Seasonal Naïve, Moving Avg
├── EVAT_Future_Forecasting_Congestion_Prediction.ipynb # SARIMAX + LSTM forecasting
├── EVAT_Congestion_Model_without...             # Alternative trial notebook
├── EVAT_future_forecasting_playbook.ipynb       # Playbook for experimentation
│
├── evat_dashboard_unified.py      # Unified Streamlit dashboard
├── evat_forecast_dashboard.py     # Forecast-focused dashboard
├── streamlit_app_3h.py            # 3-hour binning dashboard variant
│
├── arrivals_timeseries_3h.csv     # Aggregated 3h arrivals
├── history_binned.csv             # Historical demand (processed)
├── pre-processed-dataset.csv      # Final dataset for modelling
├── predictions_test_arrivals_3h.csv  # Test set predictions
├── predictions_3h_with_wait_time.csv # Forecasts + Erlang-C wait times
├── forecast_results.csv           # Model evaluation outputs
├── evaluation_metrics.json        # Evaluation results (all models)
├── evaluation_metrics_3h.json     # Evaluation (3h bin models)
├── station_data_dataverse.csv     # Raw input dataset
│
└── README.md                      # Project documentation
```

---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/EVAT-Congestion-Prediction.git
   cd EVAT-Congestion-Prediction
   ```

2. Create environment & install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Main dependencies:
   - `numpy`, `pandas`, `matplotlib`, `seaborn`
   - `scikit-learn`, `statsmodels`
   - `tensorflow` / `keras` (for LSTM)
   - `streamlit`

---

## 🚀 How to Run

### 1. Data Preprocessing
```bash
jupyter notebook "Data preprocessing.ipynb"
```

### 2. Train Models
- Baseline models:  
  ```bash
  jupyter notebook "EVAT_Congestion_with_baselines_models.ipynb"
  ```
- Advanced models (SARIMAX, LSTM):  
  ```bash
  jupyter notebook "EVAT_Future_Forecasting_Congestion_Prediction.ipynb"
  ```

### 3. Launch Dashboards
Choose one of the dashboards depending on focus:
```bash
streamlit run evat_dashboard_unified.py
streamlit run evat_forecast_dashboard.py
streamlit run streamlit_app_3h.py
```

---

## 🌍 Applications
- **Urban Planning** → identify high-demand charging stations.  
- **Operations** → inform scheduling, staffing, or pricing strategies.  
- **Policy** → evidence-based EV infrastructure investment decisions.  

---

## 📚 References
- Gross et al., *Fundamentals of Queueing Theory*, Wiley, 1998.  
- Hochreiter & Schmidhuber, *Long Short-Term Memory*, Neural Computation, 1997.  
- Box et al., *Time Series Analysis: Forecasting and Control*, Wiley, 2015.  
- International Energy Agency, *Global EV Outlook*, 2020.  

---

## 👤 Author
**Nazhim Kalam**  
Master of Data Science, Deakin University  
📧 [nazhimkalam@gmail.com](mailto:nazhimkalam@gmail.com)  
🌐 [Portfolio Website](https://nazhimkalam.netlify.app)  
