# EV Charging Station Congestion Prediction Model - Handover Documentation

**Document Version:** 1.0  
**Last Updated:** January 21, 2026  
**Model Type:** RandomForest Regressor  
**Purpose:** Real-time 3-hour arrival prediction for EV charging stations

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Model Overview](#model-overview)
3. [Technical Architecture](#technical-architecture)
4. [Data Requirements](#data-requirements)
5. [Feature Engineering](#feature-engineering)
6. [Model Performance](#model-performance)
7. [Deployment Guide](#deployment-guide)
8. [Inference Pipeline](#inference-pipeline)
9. [Maintenance & Monitoring](#maintenance--monitoring)
10. [Troubleshooting](#troubleshooting)
11. [Contact & Support](#contact--support)

---

## Executive Summary

This model predicts the number of EV arrivals at charging stations over the next 3 hours using a RandomForest regression model. The system integrates multiple data sources including temporal features, weather data, pedestrian counts, and event calendars to provide accurate real-time predictions.

**Key Capabilities:**
- Real-time predictions for multiple charging stations
- 3-hour forecast horizon
- Integration with external APIs (weather, pedestrian counts)
- Feature engineering pipeline for temporal and environmental factors
- No historical data required for inference (lag features initialized to zero)

**Use Cases:**
- Station congestion management
- Dynamic pricing strategies
- User notifications for optimal charging times
- Resource allocation planning

---

## Model Overview

### Model Type
**Algorithm:** RandomForest Regressor  
**Framework:** scikit-learn  
**Target Variable:** Number of EV arrivals in next 3 hours

### Model Artifacts
- **Model File:** `random_forest_model.pkl`
- **Inference Notebook:** `predictions_notebook.ipynb`
- **Output:** `prediction_results.csv`

### Key Characteristics
- **Input Features:** 23 engineered features
- **Stations Supported:** Currently configured for 3 stations (expandable)
- **Prediction Frequency:** On-demand (can be scheduled)
- **Response Time:** < 5 seconds per batch prediction

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Prediction Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Station IDs Input                                        │
│  2. Temporal Feature Generation (current date/time)          │
│  3. External Data Fetching:                                  │
│     - Holiday Calendar (Victoria, Australia)                 │
│     - Event Calendar (Melbourne major events)                │
│     - Weather API (Open-Meteo)                               │
│     - Pedestrian Counts (Melbourne Testbed)                  │
│  4. Feature Engineering & Transformations                    │
│  5. Model Inference (RandomForest)                           │
│  6. Results Output (CSV)                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Dependencies

**Core Libraries:**
```python
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
joblib >= 1.1.0
```

**External APIs:**
```python
requests >= 2.26.0
holidays >= 0.14  # Victoria, Australia holidays
```

**Data Sources:**
- **Weather API:** Open-Meteo Archive API (`https://archive-api.open-meteo.com/v1/archive`)
- **Pedestrian Data:** Melbourne Testbed Open Data (`https://melbournetestbed.opendatasoft.com`)

---

## Data Requirements

### Input Data

#### 1. Station IDs
```python
stations = [
    '674f97ff3dc8e5d2ac00867a',
    '674f98013dc8e5d2ac00894a',
    '674f97ff3dc8e5d2ac008456'
]
```
**Format:** List of MongoDB ObjectIds (strings)  
**Source:** Station management database

#### 2. Geographic Coordinates
**Melbourne CBD Coordinates:**
- Latitude: -37.8136
- Longitude: 144.9631

**Note:** Update coordinates if predicting for stations outside Melbourne CBD.

### External Data Sources

#### Weather Data (Open-Meteo API)
**Endpoint:** `https://archive-api.open-meteo.com/v1/archive`

**Parameters:**
```python
{
    "latitude": -37.8136,
    "longitude": 144.9631,
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,windspeed_10m_max",
    "timezone": "Australia/Melbourne"
}
```

**Retrieved Fields:**
- `temp_max_c`: Maximum temperature (°C)
- `temp_min_c`: Minimum temperature (°C)
- `temp_avg_c`: Average temperature (°C)
- `precipitation_mm`: Total precipitation (mm)
- `wind_speed_kmh`: Maximum wind speed (km/h)

**API Limits:** Free tier - check documentation for rate limits

#### Pedestrian Count Data (Melbourne Testbed)
**Endpoint:** `https://melbournetestbed.opendatasoft.com/api/explore/v2.1/catalog/datasets/pedestrian-counting-system-monthly-counts-per-hour/records`

**Query Parameters:**
```python
{
    "select": "sensing_date,hourday,direction_1",
    "where": "sensing_date >= now(days=-3) and sensing_date <= now(days=-2)",
    "timezone": "Australia/Melbourne",
    "limit": 100,
    "offset": 0  # Pagination
}
```

**Retrieved Fields:**
- `sensing_date`: Timestamp of count
- `hourday`: Hour of day (0-23)
- `direction_1`: Pedestrian count for primary direction

**Note:** Uses data from 2-3 days prior (shifted +1 day) as proxy for current pedestrian patterns

#### Holiday Calendar
**Library:** `holidays` (Python package)  
**Region:** Victoria, Australia  
**Years:** 2024-2026 (expand as needed)

**Implementation:**
```python
vic_holidays = holidays.Australia(state='VIC', years=[2024, 2025, 2026])
```

#### Event Calendar
**Major Melbourne Events:**
- Australian Open (Tennis): Jan 19-31
- AFL Season: March-September
- ANZAC Day AFL: April 25
- AFL Grand Final: Late September
- Melbourne Cup: First Tuesday of November
- Australian Grand Prix: Mid-March
- Boxing Day Test (Cricket): Dec 26-30
- New Year's Eve: Dec 31

**Function:** `categorize_event(date)` - Custom implementation

---

## Feature Engineering

### Feature Categories

The model requires **23 features** organized into 6 categories:

#### 1. Temporal Features (3 features)
| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| `hour` | Hour of day | int | 0-23 |
| `dayofweek` | Day of week | int | 0-6 (0=Monday) |
| `is_weekend` | Weekend indicator | binary | 0, 1 |

**Generation:**
```python
df['hour'] = datetime.now().hour
df['dayofweek'] = datetime.now().weekday()
df['is_weekend'] = int(datetime.now().weekday() >= 5)
```

#### 2. Lag Features (8 features)
| Feature | Description | Inference Value |
|---------|-------------|-----------------|
| `arrivals_lag1` | Previous period arrivals | 0 (no history) |
| `arrivals_lag2` | 2 periods ago | 0 |
| `arrivals_lag4` | 4 periods ago | 0 |
| `arrivals_ma4` | 4-period moving average | 0 |
| `arrivals_ma8` | 8-period moving average | 0 |
| `arrivals_pct_change` | Percentage change | 0 |
| `arrivals_diff` | First difference | 0 |
| `arrivals_ewma_4` | Exponential weighted MA | 0 |

**Note:** During inference, all lag features are initialized to **0** since no historical data is available. The model was trained to handle this scenario.

#### 3. Cyclical Temporal Features (2 features)
| Feature | Description | Formula |
|---------|-------------|---------|
| `hod_sin` | Hour of day (sine) | sin(2π × hour / 24) |
| `hod_cos` | Hour of day (cosine) | cos(2π × hour / 24) |

**Purpose:** Captures cyclical nature of time (hour 23 is close to hour 0)

#### 4. Calendar Features (2 features)
| Feature | Description | Source |
|---------|-------------|--------|
| `is_holiday` | Public holiday indicator | Victoria holidays calendar |
| `is_major_event` | Major event indicator | Custom event function |

#### 5. Weather Features (5 features)
| Feature | Unit | Source |
|---------|------|--------|
| `temp_max_c` | °C | Open-Meteo API |
| `temp_min_c` | °C | Open-Meteo API |
| `temp_avg_c` | °C | Open-Meteo API |
| `precipitation_mm` | mm | Open-Meteo API |
| `wind_speed_kmh` | km/h | Open-Meteo API |

#### 6. Foot Traffic & Interaction Features (3 features)
| Feature | Description | Formula |
|---------|-------------|---------|
| `direction_1` | Pedestrian count | Melbourne Testbed API |
| `weekend_x_hour` | Weekend-hour interaction | is_weekend × hour |
| `temp_x_precipitation` | Weather interaction | temp_avg_c × precipitation_mm |

### Feature Schema Validation

**Critical:** All 23 features must be present in exact order:
```python
required_features = [
    'hour', 'dayofweek', 'is_weekend',
    'arrivals_lag1', 'arrivals_lag2', 'arrivals_lag4',
    'arrivals_ma4', 'arrivals_ma8',
    'hod_sin', 'hod_cos',
    'is_holiday', 'is_major_event',
    'temp_max_c', 'temp_min_c', 'temp_avg_c',
    'precipitation_mm', 'wind_speed_kmh', 'direction_1',
    'arrivals_pct_change', 'arrivals_diff',
    'weekend_x_hour', 'temp_x_precipitation', 'arrivals_ewma_4'
]
```

---

## Model Performance

### Training Metrics
**Note:** Refer to training notebooks for detailed performance metrics:
- `EVAT_Congestion_Model_without_baselines.ipynb`
- `EVAT_Congestion_with_baselines_models.ipynb`

**Evaluation Files:**
- `evaluation_metrics_3h.json`
- `evaluation_metrics.json`

### Expected Outputs

**Prediction Range:** Typically 0-50 arrivals per 3-hour window  
**Output Format:** Float (continuous predictions)

**Sample Output:**
```
Station: 674f97ff3dc8e5d2ac00867a
  Predicted arrivals (3h): 12.45
  Conditions: 22.3°C, 0.0mm, pedestrians: 1250
```

### Performance Considerations

**Factors Affecting Accuracy:**
1. **Time of Day:** Better accuracy during peak hours (7-9 AM, 5-7 PM)
2. **Weather Extremes:** Lower accuracy during extreme weather events
3. **Special Events:** Predictions may vary during unscheduled major events
4. **Data Quality:** Dependent on API availability and data freshness

---

## Deployment Guide

### Prerequisites

1. **Python Environment:** Python 3.8+
2. **Model File:** `random_forest_model.pkl` in working directory
3. **Network Access:** Internet connection for API calls
4. **Geographic Scope:** Melbourne, Victoria, Australia

### Installation

```bash
# Create virtual environment
python -m venv evat_env
source evat_env/bin/activate  # Linux/Mac
# evat_env\Scripts\activate  # Windows

# Install dependencies
pip install pandas numpy scikit-learn joblib requests holidays

# Verify model file exists
ls -la random_forest_model.pkl
```

### Configuration

#### 1. Update Station IDs
Edit the stations list in Cell 7:
```python
stations = [
    'your_station_id_1',
    'your_station_id_2',
    'your_station_id_3'
]
```

#### 2. Update Geographic Coordinates (if needed)
Edit Cell 9 if stations are outside Melbourne CBD:
```python
lat = YOUR_LATITUDE
lon = YOUR_LONGITUDE
```

#### 3. Configure Event Calendar
Update `categorize_event()` function in Cell 8 for local events

### Running Predictions

#### Jupyter Notebook
```bash
jupyter notebook predictions_notebook.ipynb
# Run all cells: Cell > Run All
```

#### Python Script (if converted)
```bash
python predictions_notebook.py
```

#### Scheduled Execution
```bash
# Cron job (every 3 hours)
0 */3 * * * cd /path/to/notebook && jupyter nbconvert --to python --execute predictions_notebook.ipynb
```

### Output

**File:** `prediction_results.csv`

**Schema:**
```csv
stationid,predicted_arrivals,hour,temp_avg_c,direction_1
674f97ff3dc8e5d2ac00867a,12.45,14,22.3,1250
```

---

## Inference Pipeline

### Execution Flow

```
1. INITIALIZATION
   ├── Load model (random_forest_model.pkl)
   ├── Define station IDs
   └── Create scoring dataframe

2. TEMPORAL FEATURES
   ├── Get current timestamp
   ├── Extract hour, dayofweek, is_weekend
   └── Calculate hod_sin, hod_cos (initialized to 0)

3. LAG FEATURES
   └── Initialize all lag features to 0

4. CALENDAR DATA
   ├── Check Victoria public holidays
   └── Check major Melbourne events

5. EXTERNAL DATA FETCHING
   ├── Weather API (Open-Meteo)
   │   └── Retry on failure with exponential backoff
   └── Pedestrian API (Melbourne Testbed)
       └── Paginated retrieval (100 records/page)

6. FEATURE ENGINEERING
   ├── Merge weather data by date
   ├── Merge pedestrian data by hour
   └── Create interaction features

7. PREDICTION
   ├── Validate 23 features present
   ├── Prepare feature matrix X_score
   └── Run rf_model.predict(X_score)

8. OUTPUT
   ├── Append predictions to dataframe
   ├── Display results summary
   └── Save to prediction_results.csv
```

### Execution Time

**Typical Runtime:** 3-5 seconds
- Model loading: < 1 sec
- Feature generation: < 0.5 sec
- API calls: 1-3 sec (depends on network)
- Prediction: < 0.1 sec

---

## Maintenance & Monitoring

### Regular Maintenance Tasks

#### Daily
- [ ] Monitor API availability (weather, pedestrian data)
- [ ] Check for failed predictions in logs
- [ ] Validate output file generation

#### Weekly
- [ ] Review prediction accuracy vs. actuals
- [ ] Check for data drift in features
- [ ] Monitor API rate limits/costs

#### Monthly
- [ ] Update holiday calendar for upcoming year
- [ ] Review and update event calendar
- [ ] Analyze model performance trends
- [ ] Check for software dependency updates

#### Quarterly
- [ ] Consider model retraining with new data
- [ ] Review feature importance
- [ ] Audit data quality from external sources

### Monitoring Metrics

**Key Indicators:**
1. **Prediction Success Rate:** % of successful predictions
2. **API Uptime:** Weather and pedestrian data availability
3. **Feature Completeness:** % of predictions with all 23 features
4. **Prediction Distribution:** Monitor for anomalies

**Alert Thresholds:**
- API failures > 5% of calls
- Missing features in > 1% of predictions
- Predictions outside 0-100 range (anomaly detection)

### Model Retraining

**Trigger Conditions:**
- Significant accuracy degradation (> 20% increase in error)
- Major infrastructure changes at stations
- New data sources become available
- Seasonal pattern changes

**Retraining Process:**
1. Collect new historical data (minimum 3 months)
2. Run training notebooks with updated data
3. Evaluate new model vs. current model
4. A/B test in production
5. Replace `random_forest_model.pkl` if improvement confirmed

**Retraining Notebooks:**
- [EVAT_Congestion_Model_without_baselines.ipynb](../EVAT_Congestion_Model_without_baselines.ipynb)
- [EVAT_Congestion_with_baselines_models.ipynb](../EVAT_Congestion_with_baselines_models.ipynb)

---

## Troubleshooting

### Common Issues

#### 1. Model File Not Found
**Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'random_forest_model.pkl'`

**Solution:**
```bash
# Check current directory
pwd

# Verify model file location
find . -name "random_forest_model.pkl"

# Update path in Cell 5 if needed
rf_model = joblib.load('/full/path/to/random_forest_model.pkl')
```

#### 2. API Timeout/Failure
**Error:** `requests.exceptions.ConnectionError` or `Timeout`

**Solutions:**
- Check internet connectivity
- Verify API endpoints are accessible
- Implement retry logic:
```python
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(total=3, backoff_factor=1)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)
```

#### 3. Missing Features
**Error:** `⚠ Missing: ['feature_name']`

**Solution:**
- Check data fetching steps completed successfully
- Verify API responses contain expected fields
- Check for column name mismatches
- Review merge operations (ensure `how='left'` preserves all rows)

#### 4. Incorrect Predictions
**Issue:** Predictions seem unrealistic (negative, extremely high, or zero)

**Debugging Steps:**
1. Verify feature values are in expected ranges
2. Check for NaN/Null values in features
3. Validate model loaded correctly (check n_estimators, max_depth)
4. Review input data for anomalies
5. Compare current conditions to training data distribution

#### 5. Pedestrian Data Not Found
**Error:** No pedestrian records retrieved

**Solutions:**
- Adjust date range in query (`now(days=-X)`)
- Check Melbourne Testbed API status
- Verify timezone parameter is correct
- Implement fallback: use average pedestrian count

#### 6. Date/Time Issues
**Issue:** Incorrect temporal features or timezone problems

**Solutions:**
```python
# Ensure timezone-aware datetime
from datetime import datetime
import pytz

melbourne_tz = pytz.timezone('Australia/Melbourne')
current_time = datetime.now(melbourne_tz)
```

### Logging & Debugging

**Add logging to track execution:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='predictions.log'
)

logging.info(f"Starting prediction for {len(stations)} stations")
logging.info(f"Features generated: {df_score.columns.tolist()}")
logging.info(f"Predictions completed: {predictions}")
```

### Emergency Contacts

**API Support:**
- Open-Meteo: https://open-meteo.com/en/docs
- Melbourne Testbed: https://melbournetestbed.opendatasoft.com/

**Internal Contacts:**
- Data Science Team: [contact information]
- DevOps/Infrastructure: [contact information]
- Product Owner: [contact information]

---

## Contact & Support

### Documentation
- **Training Notebooks:** `congestion-forecasting/` directory
- **Model Artifacts:** `congestion-forecasting/artifacts_premium/`
- **Dashboard Code:** `evat_dashboard_unified.py`, `streamlit_app_3h.py`
- **API Documentation:** `EVAT-App-BE.postman_collection.json`

### Project Structure
```
EVAT-Data-Science/
├── congestion-forecasting/
│   ├── EVAT_Congestion_Model_without_baselines.ipynb  # Training
│   ├── EVAT_Congestion_with_baselines_models.ipynb    # Training
│   ├── artifacts_premium/
│   │   ├── poisson_lstm_best.keras
│   │   ├── predictions_3h_with_wait_times.csv
│   │   └── evaluation_metrics_3h.json
│   └── Use_Cases/Congestion Prediction/Prediction/
│       ├── predictions_notebook.ipynb                 # INFERENCE (THIS NOTEBOOK)
│       ├── random_forest_model.pkl                    # MODEL FILE
│       └── MODEL_DOCUMENTATION.md                     # THIS FILE
```

### Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-01-21 | Initial documentation | EVAT Data Science Team |

### Future Enhancements

**Planned Improvements:**
1. Real-time historical data integration for lag features
2. Multi-step ahead forecasting (6h, 12h, 24h)
3. Station clustering for location-based features
4. Real-time model updates with online learning
5. Integration with queueing theory models
6. Mobile API endpoint development

### Feedback & Contributions

For questions, issues, or suggestions regarding this model:
1. Review this documentation thoroughly
2. Check training notebooks for additional context
3. Contact the Data Science team
4. Submit issues/PRs to project repository

---

## Appendix

### A. Complete Feature List with Descriptions

| # | Feature Name | Category | Description | Source |
|---|--------------|----------|-------------|--------|
| 1 | hour | Temporal | Hour of day (0-23) | System clock |
| 2 | dayofweek | Temporal | Day of week (0=Mon, 6=Sun) | System clock |
| 3 | is_weekend | Temporal | Weekend indicator (Sat/Sun) | Derived |
| 4 | arrivals_lag1 | Lag | Previous period arrivals | Historical (0 in inference) |
| 5 | arrivals_lag2 | Lag | 2 periods back | Historical (0 in inference) |
| 6 | arrivals_lag4 | Lag | 4 periods back | Historical (0 in inference) |
| 7 | arrivals_ma4 | Lag | 4-period moving average | Historical (0 in inference) |
| 8 | arrivals_ma8 | Lag | 8-period moving average | Historical (0 in inference) |
| 9 | hod_sin | Cyclical | Sine of hour | Derived |
| 10 | hod_cos | Cyclical | Cosine of hour | Derived |
| 11 | is_holiday | Calendar | Public holiday indicator | Holidays library |
| 12 | is_major_event | Calendar | Major event indicator | Custom function |
| 13 | temp_max_c | Weather | Daily max temperature (°C) | Open-Meteo API |
| 14 | temp_min_c | Weather | Daily min temperature (°C) | Open-Meteo API |
| 15 | temp_avg_c | Weather | Daily avg temperature (°C) | Open-Meteo API |
| 16 | precipitation_mm | Weather | Daily precipitation (mm) | Open-Meteo API |
| 17 | wind_speed_kmh | Weather | Max wind speed (km/h) | Open-Meteo API |
| 18 | direction_1 | Foot Traffic | Pedestrian count | Melbourne Testbed |
| 19 | arrivals_pct_change | Lag | % change from previous | Historical (0 in inference) |
| 20 | arrivals_diff | Lag | First difference | Historical (0 in inference) |
| 21 | weekend_x_hour | Interaction | Weekend × Hour | Derived |
| 22 | temp_x_precipitation | Interaction | Temperature × Rain | Derived |
| 23 | arrivals_ewma_4 | Lag | Exponential weighted MA | Historical (0 in inference) |

### B. Sample API Responses

**Weather API Response:**
```json
{
  "daily": {
    "time": ["2026-01-21"],
    "temperature_2m_max": [28.5],
    "temperature_2m_min": [18.2],
    "temperature_2m_mean": [23.1],
    "precipitation_sum": [0.0],
    "windspeed_10m_max": [15.3]
  }
}
```

**Pedestrian API Response:**
```json
{
  "results": [
    {
      "sensing_date": "2026-01-19T14:00:00+11:00",
      "hourday": 14,
      "direction_1": 1250
    }
  ]
}
```

### C. Glossary

- **EV:** Electric Vehicle
- **Lag Feature:** Historical value from previous time periods
- **Moving Average (MA):** Average over sliding window of time periods
- **EWMA:** Exponentially Weighted Moving Average (recent values weighted more)
- **RandomForest:** Ensemble learning method using multiple decision trees
- **Feature Engineering:** Creating new input variables from raw data
- **Inference:** Making predictions using a trained model
- **3h Window:** 3-hour time period for prediction horizon

---

**End of Documentation**

*For the latest version of this document, refer to the project repository.*
