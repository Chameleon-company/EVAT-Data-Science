# EV Charging Trip Prediction using Machine Learning

This project applies supervised machine learning models to predict electric vehicle (EV) charging trip characteristics, including estimated time of arrival (ETA) and the destination charging station.

## Dataset
`ml_ev_charging_dataset.csv`

Features include:
- Geospatial coordinates (vehicle and station)
- Categorical station identifiers
- Calculated Haversine distances

---

##  Project Objectives

1. **Regression** – Predict ETA (in minutes) using geospatial and categorical features.
2. **Binary Classification** – Categorize ETA as `Short` (≤ 5 min) or `Long`.
3. **Multi-Class Classification** – Predict the specific EV charging `Station_Name`.

---

##  Models Used

- **Linear & Logistic Regression**
- **Random Forest (Regressor & Classifier)**
- **XGBoost (Regressor & Classifier)**

---

## Key Results

###  Regression (`ETA_min`)
| Model | MAE | RMSE |
|-------|-----|------|
| Linear Regression | 1.19 | 1.53 |
| Random Forest     | 0.47 | 0.74 |
| XGBoost           | **0.32** | **0.60** |

###  Binary Classification (`ETA_Class`)
| Model | Accuracy | F1 (Short) | F1 (Long) |
|-------|----------|------------|-----------|
| Logistic Regression | 92.86% | 0.95 | 0.88 |
| Random Forest       | 97.32% | 0.98 | 0.95 |
| XGBoost             | **97.32%** | **0.98** | **0.95** |

###  Multi-Class Classification (`Station_Name`)
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 87.5% |
| Random Forest       | **100%** |
| XGBoost             | **100%** |

---

## Recommendations

- **XGBoost** is the best-performing model overall.
- **Random Forest** is a reliable alternative with nearly equal performance.
- Address class imbalance for logistic regression with SMOTE or class weighting.

---

## Future Work

- Hyperparameter tuning (e.g., with GridSearchCV) to further optimize model performance.
- Address class imbalance using **SMOTE** or **class weighting** to improve generalization.
- Explore **feature interactions** and **non-linear transformations** to support linear models.
- Integrate models into a real-time **dashboard or API** (e.g., with Flask or Streamlit).
- **Increase dataset size** – The current dataset includes ~600 rows; incorporating more data will improve model robustness, reduce variance, and allow for proper **time-series modeling** to better capture temporal dynamics in EV charging behavior.
