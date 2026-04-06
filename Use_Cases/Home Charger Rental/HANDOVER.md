# Home Charger Rental – Handover Notes (Sprint 3)

This document summarises the work completed for the **Home Charger Rental** use case and provides clear handover guidance for other team members (API/Web/ML).

Maintained by: **Daniel Nguyen**

---

## 1. Scope of work completed

### Data preparation (cleaning & organisation)
- Cleaned and standardised multiple datasets related to suburb-level EV readiness.
- Organised cleaned outputs into a dedicated folder:
  - `datasets/clean/`
- Ensured consistent suburb matching by creating a standard suburb key (`suburb_key`).

### Master table creation
- Merged cleaned datasets into a single suburb-level master table to support modelling and API handover:
  - `datasets/clean/master_suburb_table.csv`
  - Updated version(s) may include congestion and additional derived metrics:
  - `datasets/clean/master_suburb_table_with_congestion.csv`

### Derived metrics
- Created simple, interpretable suburb-level indicators:
  - Charger-to-population ratio
  - Station-to-population ratio
  - Congestion proxy (filled using median where missing)

### Clustering interpretation
- Built / used clustering outputs to segment suburbs and provide business-readable labels:
  - Example labels:
    - **EV-Ready Suburbs**
    - **High Population – Infrastructure Gap**
    - **Affluent Car-Dependent (High Potential)**

### Pricing signal
- Produced suburb-level indicative pricing output:
  - `datasets/clean/optimal_pricing_by_suburb.csv`
- This is an analytical signal intended to support decision-making, not a final pricing engine.

---

## 2. Folder structure (current)

Recommended structure used during analysis:

Use_Cases/Home Charger Rental/
- Analysis/
  - notebooks for cleaning, clustering, pricing
- api/
  - openapi.yaml
  - README.md
- datasets/
  - clean/
    - master_suburb_table*.csv
    - optimal_pricing_by_suburb.csv
    - charger_info_mel_clean.csv
    - info_for_pcz_clean.csv
    - stations_per_town_clean.csv
    - suburb_population_clean.csv
    - coordinates_clean.csv (if available)
    - other intermediate clean files

---

## 3. Key outputs for integration

### A) Master suburb table (recommended for API)
Suggested primary file for integration:
- `datasets/clean/master_suburb_table_with_congestion.csv`

Contains:
- Population, dwellings, income, vehicles per dwelling
- Public charger count, station count
- Ratios (charger/station per population)
- Congestion proxy (Avg_Congestion)
- Cluster (if joined in final step)

### B) Pricing output
- `datasets/clean/optimal_pricing_by_suburb.csv`

Contains:
- `Suburb`
- `Cluster_Label`
- `Optimal_Price`

---

## 4. Known limitations / data caveats

- Coordinates coverage may be limited depending on the dataset provided.
- Congestion values may be missing for many suburbs; median fill is used for completeness.
- Pricing output is a simplified analytical model and should be treated as indicative.

---

## 5. API handover (alignment document)

- OpenAPI specification is stored at:
  - `api/openapi.yaml`
- Companion explanation for non-technical readers:
  - `api/README.md`

Suggested minimal response fields for the EVAT app:
- `suburb`
- `cluster_id`
- `cluster_label`
- `metrics` (population, chargers, stations, ratios, congestion)
- `optimal_price`

---

## 6. Recommended next steps (for team)

### API Team
- Confirm final API endpoints and minimal stable response fields.
- Implement a lightweight service (FastAPI recommended) that reads from cleaned CSV outputs.
- Add input validation for suburb name matching.

### Web/UI Team
- Confirm how clusters and pricing signals will be visualised (map/list/cards).
- Decide which metrics to show by default and which are optional.

### Data/ML Team
- Improve coordinate coverage (optional):
  - derive suburb centroid via charger coordinates or external suburb boundary datasets
- Revisit congestion feature:
  - validate whether the congestion metric aligns with EV charging demand behaviour

---