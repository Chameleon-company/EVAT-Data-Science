# CO₂ Savings Prediction API – Input/Output Schema

This document defines the required input fields, data types, and response structure for the **Environmental Impact Analysis (CO₂ Savings Prediction)** API.  

---

## 1. API Overview

The API predicts how many grams of CO₂ per kilometer are saved when replacing an **Internal Combustion Engine (ICE)** vehicle with an **Electric Vehicle (EV)**.

Predictions are based on:

- EV energy consumption  
- ICE fuel-based CO₂ baseline  
- Vehicle body style  
- Make/brand  
- Vehicle release year difference  

---

## 2. Input Schema (JSON Request)

Send a single JSON object with the following fields:

| Field                | Type    | Description                                                   |
|----------------------|---------|---------------------------------------------------------------|
| **Make_EV**          | string  | Brand of the EV (e.g., "Tesla")                              |
| **Make_ICE**         | string  | Brand of the ICE vehicle (e.g., "Toyota")                    |
| **BodyStyle_EV**     | string  | EV body style ("SUV", "Sedan", etc.)                         |
| **BodyStyle_ICE**    | string  | ICE body style                                               |
| **FuelType_ICE**     | string  | One of: "Petrol91", "Petrol95", "Petrol98", "Diesel"         |
| **YearDiff**         | number  | EV model year minus ICE model year                           |
| **ICE_CO2_Baseline** | number  | Pre-calculated CO₂ emissions of the ICE vehicle (g/km)       |

### Example Input JSON

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

## 3. Output Schema (JSON Response)

The API returns the estimated CO₂ savings in grams per km.

| Field                     | Type  | Description                                        |
|---------------------------|-------|----------------------------------------------------|
| **Predicted_CO2_Savings** | float | Estimated CO₂ saved when switching from ICE → EV   |

### Example Output JSON

```json
{
  "Predicted_CO2_Savings": 134.72
}
```

This means the EV is predicted to emit **134.72 g/km less CO₂** than the ICE vehicle.

---

## 4. Validation Rules & Constraints

- All string values must match the categories used during model training  
- Unseen car brands may reduce accuracy  
- `YearDiff` must be numeric (positive or negative)  
- `ICE_CO2_Baseline` must be pre-computed before calling the API  
- Missing or incorrectly spelled fields will cause prediction errors  

---

## 5. Notes for Backend Developers

- The API processes **one EV–ICE pair at a time**  
- For batch predictions, run a loop on the backend  
- Ensure correct casing and spelling ("Petrol95" not "petrol95")  
- Accepted fuel types: **Petrol91, Petrol95, Petrol98, Diesel**  

---
