# Home Charger Rental – Data & Web Prototype

This project explores the **Home EV Charger Rental use case** by combining **data-driven clustering, demand simulation, neural network pricing optimization**, and a **basic booking/listing web interface**.

---

## Use Case Overview

Many urban and suburban households own EV chargers that remain unused during the day. By renting them out, hosts can generate income while drivers get more accessible charging options.
This repository provides:

1. **Clustering Pipeline** → Groups suburbs by EV-readiness, population, congestion, and infrastructure gaps.
2. **Usage Simulation & Neural Network Model** → Generates synthetic rental demand, predicts usage, and finds **optimal dynamic pricing**.
3. **Web Prototype** → A basic **Booking** (for users) & **Listing** (for hosts) HTML interface to demonstrate how the service could work.

---

## 📂 Repository Structure

```
├── ev_charging_clustering_pipeline.py   # Clusters suburbs and prepares inputs
├── ev_usage_simulation_nn.py            # Simulates demand + trains NN for optimal pricing
├── datasets/
│   ├── clustered_suburbs.csv            # Clustered dataset (output)
│   ├── optimal_prices_all_suburbs.csv   # NN-based optimal prices (output)
│   ├── Charger-Info.csv                 # Charger availability & metadata
│   ├── vehicle_registrations.csv        # EV adoption by suburb/region
│   ├── ml_ev_charging_dataset.csv       # Travel behavior + EV usage dataset
│   ├── road_congestion.csv              # Road congestion levels by suburb
│   ├── Suburb_Population.csv            # Population by suburb
│   ├── Info_for_PCZ.csv                 # Dwelling & zoning information
│   └── stations_per_town.csv            # Public charging stations per town
├── UI_Interface/
│   ├── booking.html                     # Booking page prototype
│   └── listing.html                     # Listing page prototype
└── README.md

```

---

## Clone the repository

```bash
git clone https://github.com/Sehar-Aejaz/home-charger-rental.git
cd home-charger-rental
```

Minimal requirements:

* `pandas`
* `numpy`
* `scikit-learn`
* `torch`
* `matplotlib`

---

## Pipelines

### 1. **EV Charging Clustering**

File: `ev_charging_clustering_pipeline.py`

* Loads suburb-level population, congestion, and EV adoption data.
* Runs **KMeans clustering** to group suburbs into categories:

  * **Urban EV-Ready**
  * **Growth Potential**
  * **Infrastructure Gap**
* Saves clustered dataset → `datasets/clustered_suburbs.csv`.

---

### 2. **Demand Simulation & Neural Network Optimization**

File: `ev_usage_simulation_pricing.py`

* Simulates **6 months of daily demand** across suburbs.
* Trains a **neural network** to predict rental usage.
* Implements **optimal pricing function**:

  * Finds the best price per suburb that maximizes expected revenue.
* Saves results → `datasets/optimal_prices_all_suburbs.csv`.

---

### 3. **Web Prototype**

Folder: `UI_Interface/`

* `listing.html` → Hosts can create a charger listing (suburb, price, availability).
* `booking.html` → Renters can view chargers and book.
  This is a **static prototype** – no backend is connected yet.

---

## Example Outputs

* `simulated_rental_usage.csv` → Daily suburb-level rental demand.
* `optimal_prices_all_suburbs.csv` → Optimal dynamic prices per suburb.

---

## Next Steps

* Connect the **pricing model outputs** to the **web app backend**.
* Add **real payment & booking flows**.
* Use **real EV charging datasets** to validate clusters & predictions.
* Explore **reinforcement learning** for dynamic pricing.

