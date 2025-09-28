# Weather-Aware Routing – EVAT Project

## Project Overview
This repository contains the work completed as part of the EVAT (Electric Vehicle Adoption Tools) Data Science team.  
The project focuses on **Weather-Aware Routing**, where traffic, weather, and EV charging datasets were collected, combined, and analysed to support intelligent routing decisions for electric vehicles.

---

## Sprint Work Completed

### **Sprint 1 – Data Collection and Integration**
- Collected **weather, traffic, and EV charging datasets** for Melbourne.
- Cleaned, standardised, and merged them into a single combined dataset.
- Engineered new features like **temperature range** for further analysis.

### **Sprint 2 – Preprocessing and Scoring**
- Designed a **Weather Sensitivity Scoring Logic** using temperature, precipitation, and weather extremes.
- Normalised values to produce a score between 0 and 1 for use in routing.
- Extended datasets (e.g., expanded EV charger dataset to 300 rows) to ensure robustness in analysis.

### **Sprint 3 – Machine Learning Models**
- Built and tested a **Random Forest regression model** to predict weather sensitivity scores.
- Achieved strong results (R² ~ 1.0), validating preprocessing and feature engineering.
- Analysed feature importance and confirmed predictive value of weather and traffic variables.

### **Sprint 4 – Dynamic Routing Prototype**
- Implemented a prototype **routing engine** that adjusts paths based on weather and traffic conditions.
- Built logic to calculate **weighted route scores** combining distance, weather sensitivity, and congestion.
- Created **preview HTML maps** showing dynamic EV routes under different conditions.
- Began integration with an app-style structure (`engine`, `adapters`, `ui_app`, etc.), preparing for future API endpoints.

---

## Files in Repository
- **Sprint 1–2**: Dataset preprocessing and Weather Sensitivity Scoring (`Weather_Aware_Routing.ipynb`)
- **Sprint 3**: Random Forest model and feature analysis
- **Sprint 4**: Dynamic routing engine prototype (`engine.py`, `adapters.py`, `ui_app.py`, `utils.py`) and interactive HTML maps (`evat_route_preview.html`)

---

## Purpose
This project demonstrates how **multi-source data integration (traffic, weather, chargers)** can be used to improve EV route planning. The goal is to enable **dynamic, weather-aware routing** that helps EV drivers make efficient and safe travel decisions.
