# Data Sources

This folder contains four CSV files that power the Environmental Impact Analysis model.
Two are embedded (created from authoritative Australian publications), two are sourced from
the Green Vehicle Guide (Australia's official vehicle emissions database).

---

## Files in this folder

| File | Rows | Description |
|------|------|-------------|
| `grid_emission_factors.csv` | 8 | State-level grid intensity, DCCEEW 2023 |
| `fuel_emission_factors.csv` | 6 | Fuel type CO2 factors, DCCEEW 2023 |
| `ev_vehicles.csv` | 58 | EV models with WLTP energy consumption |
| `ice_vehicles.csv` | 65 | ICE vehicles with WLTP fuel consumption |

---

## Data Sources & References

### 1. Grid Emission Factors (`grid_emission_factors.csv`)
**Source:** DCCEEW National Greenhouse Accounts Factors 2023, Table 3  
**URL:** https://www.dcceew.gov.au/climate-change/publications/national-greenhouse-accounts-factors  
**Status:** Embedded in this repo — values are static annual factors.  
**Update frequency:** Annually (new edition published ~August each year).

These are **Scope 2, location-based** emission factors for grid electricity consumption.
Use these to convert EV electricity consumption (kWh) into CO2-equivalent emissions.

| State | g CO2/kWh | Primary generation |
|-------|-----------|-------------------|
| NSW   | 790       | Black coal + gas |
| ACT   | 0         | 100% renewable purchase |
| VIC   | 990       | Brown coal (lignite) |
| QLD   | 810       | Black coal |
| SA    | 290       | Wind + solar |
| WA    | 650       | Gas (SWIS grid) |
| TAS   | 130       | Hydro |
| NT    | 590       | Gas (isolated) |

**To update:** Download the latest NGA Factors PDF, find Table 3, update the CSV.

---

### 2. Fuel Emission Factors (`fuel_emission_factors.csv`)
**Source:** DCCEEW National Greenhouse Accounts Factors 2023, Table 1  
**URL:** https://www.dcceew.gov.au/climate-change/publications/national-greenhouse-accounts-factors  
**Status:** Embedded — static Scope 1 combustion factors.

These are used as: `CO2_g_per_km = (L/100km × emission_factor_kg/L × 1000) / 100`

---

### 3. EV Vehicle Database (`ev_vehicles.csv`)
**Source:** Green Vehicle Guide (Australian Government)  
**URL:** https://www.greenvehicleguide.gov.au/  
**Status:** Manually compiled from GVG search results (current as of April 2025).  
**Update frequency:** GVG is updated as new vehicles receive approval.

**To expand this dataset:**
1. Go to https://www.greenvehicleguide.gov.au/
2. Set "Fuel type" to "Electric"
3. Export or manually record:
   - Make, Model, Year, Variant
   - Energy Consumption (Wh/km) — this is the WLTP AC socket figure
   - Battery capacity (kWh) — from manufacturer specs if not on GVG
4. Add rows to `ev_vehicles.csv` following the existing schema

**Key column:** `Consumption_kWh_per_100km` — divide Wh/km by 10 to get kWh/100km

---

### 4. ICE Vehicle Database (`ice_vehicles.csv`)
**Source:** Green Vehicle Guide (Australian Government)  
**URL:** https://www.greenvehicleguide.gov.au/  
**Status:** Manually compiled from GVG search results (current as of April 2025).

**To expand this dataset:**
1. Go to https://www.greenvehicleguide.gov.au/
2. Set "Fuel type" to Petrol / Diesel / LPG as needed
3. Export or manually record:
   - Make, Model, Year, Variant, FuelType
   - Combined fuel consumption (L/100km)
   - CO2 combined (g/km) — for cross-validation only; model computes this itself
4. Add rows following the existing schema

---

## Why not use a direct API?

The Green Vehicle Guide does not offer a public REST API. The DCCEEW NGA Factors are
published as a PDF/Excel document rather than a machine-readable endpoint.

**Future improvement:** If the backend team can set up a scheduled job to scrape GVG
and store vehicle data in the EVAT database, the CSV lookup in `data_loader.py` can be
swapped for a database query with zero changes to `calculator.py` or `api/main.py`.

---

## Extending coverage

If a user selects a vehicle not in the database, the API returns a 404 with a helpful
message asking them to provide `consumption_kwh_per_100km` and `battery_kwh` manually.
This allows any vehicle to be compared even without a database entry.
