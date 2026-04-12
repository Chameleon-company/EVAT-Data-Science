# Personalised EV Usage Insights Documentation

**Document Version:** 1.0  
**Last Updated:** April 2026  
**System Type:** Data Pipeline + Clustering + Dashboard  
**Purpose:** Provide personalised EV insights based on user driving behaviour  

---

##  Table of Contents

1. [Executive Summary](#1-executive-summary)  
2. [System Overview](#2-system-overview)  
3. [Technical Architecture](#3-technical-architecture)  
4. [Local Setup & Execution](#4-local-setup--execution)  
   - 4.1 [Backend Setup](#41-backend-setup-nodejs)  
   - 4.2 [Clustering Service Setup](#42-clustering-service-setup-python)  
   - 4.3 [Environment Configuration](#43-environment-configuration)  
5. [API Testing](#5-api-testing)  
6. [Machine Learning Integration](#6-machine-learning-integration)  
7. [Database Structure](#7-database-structure)  
8. [Dashboard (Power BI)](#8-dashboard-power-bi)  
9. [Important Observations](#9-important-observations)  
10. [System Workflow](#10-system-workflow)  
11. [Limitations](#11-limitations)  
12. [Conclusion](#12-conclusion)  

---

## 1. Executive Summary

The Personalised EV Usage Insights system analyses user driving patterns and provides insights into fuel consumption, cost savings, and EV adoption potential.

The system integrates:
- Data collection (user inputs)
- Backend processing (Node.js API)
- Machine learning clustering (Python Flask service)
- Data storage (MongoDB)
- Visualisation (Power BI dashboard)

---

## 2. System Overview

### Key Capabilities

- Accepts user driving data via API  
- Assigns users to behavioural clusters  
- Stores processed data in MongoDB  
- Provides EV savings insights  
- Visualises insights through Power BI  

---

## 3. Technical Architecture
User Input (Thunder Client / Form)
        ↓
Node.js Backend (Webhook API)
        ↓
Python Flask Service (Clustering)
        ↓
MongoDB (Data Storage)
        ↓
Power BI Dashboard (Visualisation)


---

## 4. Local Setup & Execution

### 4.1 Backend Setup (Node.js)
Navigate to `evat_webhook/`, then run:
```bash
npm install
node index.js
```
Server runs on: `http://localhost:3000`

### 4.2 Clustering Service Setup (Python)
Navigate to `evat_cluster_service/`, then run:
```bash
pip install -r requirements.txt
python app.py
```
Service runs on: `http://127.0.0.1:8000`

### 4.3 Environment Configuration
`.env` file:
```env
MONGO_URI=your_mongodb_connection
PORT=3000
PREDICT_URL=http://127.0.0.1:8000/predict
```

---

## 5. API Testing

**Endpoint:** `POST /api/save`

**Sample Input:**
```json
{
  "weekly_km": 250,
  "fuel_efficiency": 8.5,
  "monthly_fuel_spend": 320,
  "trip_length": "Medium",
  "driving_frequency": "Daily",
  "driving_type": "City",
  "road_trips": "Occasionally",
  "car_ownership": "Own",
  "home_charging": "No",
  "solar_panels": "No",
  "charging_preference": "Home",
  "budget": "Medium",
  "priorities": "Cost Savings",
  "postcode": "3000",
  "email": "test@example.com"
}
```

**Observed Behaviour:**  Request received →  Sent to clustering →  Cluster assigned →  Stored in MongoDB

---

## 6. Machine Learning Integration
-  Algorithm: K-Prototypes
-  Handles numerical and categorical data
-  Generates 4 user segments
-  Flask API exposes `/predict` endpoint

---

## 7. Database Structure

**Database:** `EVAT` | **Collection:** `user_responses`

```json
{
  "weekly_km": 93,
  "fuel_efficiency": 8.5,
  "monthly_fuel_spend": 51.96,
  "driving_type": "City",
  "email": "test@example.com",
  "cluster": 3
}
```

---

## 8. Dashboard (Power BI)
Provides:  Driving statistics ·  Cluster segmentation ·  EV savings estimation ·  Driver comparisons

**Key Visuals:** Fuel efficiency · Monthly spend · Weekly distance · Profile classification

---

## 9. Important Observations
-  Backend and clustering work in real-time
-  MongoDB stores processed user data
-  Power BI uses a preloaded dataset (not live)
-  Dashboard reflects expected analytical insights

---

## 10. System Workflow
```
User submits data → Backend processes → Clustering assigns segment → MongoDB stores → Dashboard visualises
```

---

## 11. Limitations
-  Power BI not connected to live MongoDB
-  Authentication not fully tested
-  Dashboard uses static dataset
-  Real-time dashboard integration not implemented

---

## 12. Conclusion
The system successfully integrates backend APIs, machine learning clustering, and data visualisation for personalised EV insights.

 End-to-end data flow works ·  Clustering correctly applied ·  Insights logically represented
