# EDA Summary – Home Charger Rental Use Case

## 1. Dataset Overview
The main dataset used for exploratory data analysis is:

**`ml_ev_charging_dataset.csv`**
- **Rows:** 558  
- **Columns:** 9  

| Column | Type | Description |
|--------|------|-------------|
| Timestamp | object | timestamp of the query/search event |
| Station_Name | object | charging provider (bp, JOLT, Evie, ChargeFox, etc.) |
| Longitude | float64 | station longitude |
| Latitude | float64 | station latitude |
| Address | object | station address |
| Distance_km | float64 | distance from suburb to station |
| ETA_min | float64 | estimated travel time in minutes |
| Suburb_Location_Lat | float64 | user's suburb latitude |
| Suburb_Location_Lon | float64 | user's suburb longitude |

This dataset reflects real or simulated user queries when searching for EV charging stations in Greater Melbourne.

---

## 2. Data Quality Assessment

### Missing Values
All columns have **0% missing values**.

➡️ The dataset is **clean** and does not require imputation or missing value handling.

### Data Types
- 3 object columns  
- 6 numeric columns  

➡️ Consistent and ready for analysis/modeling.

---

## 3. Statistical Summary

| Feature | Mean | Std | Min | Max | Notes |
|--------|------|-----|-----|-----|-------|
| Distance_km | 4.13 | 3.33 | 0.26 | 14.47 | Right-skewed; most values 1–5 km |
| ETA_min | 7.55 | 3.91 | 1.5 | 21.0 | Similar shape to Distance_km |
| Longitude | 144.98 | 0.21 | 144.57 | 145.61 | Within Melbourne region |
| Latitude | -37.81 | 0.15 | -38.37 | -37.51 | Within Melbourne region |

---

## 4. Distribution Insights

### Longitude & Latitude
- Follow near-normal distributions.
- No major outliers.
- Indicates consistent station locations across Melbourne.

### Distance_km
- Strong right skew.
- Most users are within **1–5 km** of a station.
- A few cases above 10 km may indicate low charger coverage areas.

### ETA_min
- Mirrors the distribution of Distance_km.
- Majority of travel times fall below 10 minutes.

### Suburb Location Coordinates
- Distribution matches station coordinates.
- Clean and free of abnormal values.

---

## 5. Correlation Analysis

### Strong Correlations
| Pair | Correlation | Interpretation |
|------|-------------|----------------|
| Distance_km ↔ ETA_min | **0.93** | Distance is the dominant factor for ETA |
| Longitude ↔ Suburb_Location_Lon | **0.99** | Station and suburb longitudes are almost identical |
| Latitude ↔ Suburb_Location_Lat | **0.99** | Station and suburb latitudes are almost identical |

### Key Takeaways
- Coordinates of suburb and station are highly redundant.
- Only one coordinate set should be kept to avoid multicollinearity.
- ETA_min and Distance_km contain overlapping information.

💡 Suggested engineered feature: travel_speed = Distance_km / ETA_min


### Weaker Correlations
- All other pairings show low or moderate correlation.
- No significant multicollinearity beyond coordinate duplication.

---

## 6. Feature Engineering Recommendations
To enhance modeling:

- `travel_speed` (captures traffic/route quality)
- `is_peak_hour` (extract hour from Timestamp)
- One-hot encoding for `Station_Name`
- Removal of redundant coordinate features

---

## 7. Key Insights Summary

- Dataset is clean and contains no missing values.
- Distribution of features is logical and aligned with expected user behavior.
- Strong correlation between suburb and station coordinates → remove duplicates.
- Strong dependency between distance and ETA → reduce dimensionality or engineer new features.
- Dataset is fully suitable for modeling tasks such as recommendation or clustering.

---

## 8. Next Steps (Sprint Ready)
- Select final feature set for modeling.
- Check other datasets (clustered_suburbs, charger_info_mel).
- Build a baseline recommendation model (nearest station or shortest ETA).

