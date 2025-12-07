# Environmental Impact Analysis — CO₂ Savings Prediction

This use case predicts how much CO₂ emissions can be saved when replacing an Internal Combustion Engine (ICE) vehicle with an Electric Vehicle (EV).  
The model compares EV energy consumption with ICE fuel consumption and estimates grams of CO₂ saved per km.

A positive CO₂ savings value means **the EV emits less CO₂** than the ICE vehicle.

---

## 1. Data Sources  
The model uses five datasets:

| Dataset | Description |
|--------|-------------|
| Pure electric consumption.csv | EV energy usage (Wh/km) |
| Diesel ICE fuel usage.csv | Diesel ICE consumption |
| petrol91RON consumption.csv | Petrol 91 ICE consumption |
| petrol95RON consumption.csv | Petrol 95 ICE consumption |
| petrol98RON consumption.csv | Petrol 98 ICE consumption |

---

## 2. Feature Engineering  
Key engineered features:

| Feature | Meaning |
|--------|---------|
| EV_gCO2_per_km | EV emissions converted from Wh/km → gCO₂/km |
| ICE_CO2_Baseline | Estimated CO₂ for each ICE vehicle (fuel × emission factor) |
| YearDiff | EV model year – ICE model year |
| CO2_saving | ICE_CO2_Baseline – EV_gCO2_per_km |

Emission factors:
- Petrol: **23.2 kg CO₂ per L**  
- Diesel: **26.5 kg CO₂ per L**  
- Electricity: **0.18 kg CO₂ per kWh**

---

## 3. Modeling Approach  
Tested three models with 5-fold cross-validation:

| Model | Mean R² | Mean MAE | Notes |
|-------|---------|----------|-------|
| Linear Regression | Low | Moderate | Underfitting |
| Random Forest | Moderate | Good | Stable |
| Gradient Boosting | **Best** | **Lowest error** | Selected model |

**Final chosen model:** GradientBoostingRegressor

---

## 4. Production Model  
The production pipeline:

- Loads and cleans data  
- Combines EV + ICE using Cartesian merge  
- Encodes categorical variables (OneHotEncoder)  
- Builds preprocessing + model pipeline  
- Saves trained model as: **co2_savings_model.pkl**

---

## 5. Input Schema (API request)

Example JSON input:

```json
{
  "Make_EV": "Tesla",
  "Make_ICE": "Toyota",
  "BodyStyle_EV": "SUV",
  "BodyStyle_ICE": "SUV",
  "FuelType_ICE": "Petrol95",
  "YearDiff": 5,
  "ICE_CO2_Baseline": 220.4
}
```

---

## 6. Output Schema (API response)

Example:

```json
{
  "Predicted_CO2_Savings": 134.72
}
```

This means the EV is predicted to emit **134.72 g/km less CO₂** than the ICE vehicle.

---

## 7. FastAPI Endpoint (Draft)

```
POST /predict
```

**Description:**  
Accepts JSON input and returns the estimated CO₂ savings.

---

## 8. How the App Team Will Use This

The mobile/web application will:

1. Send EV + ICE vehicle details to the API  
2. Receive the predicted CO₂ savings  
3. Display environmental benefits and comparisons to the user  

---
