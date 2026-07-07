# Home Charger Rental – Analysis Handover Notes

## 1) Purpose
This folder documents the data review/cleaning, suburb-level clustering, and indicative pricing signals for the Home Charger Rental use case. The goal is to produce decision-support outputs (cluster summaries + pricing signals) that can be exposed to the EVAT app via a simple API.

---

## 2) Key Notebooks
- `01_data_review_cleaning.ipynb`  
  Data review and cleaning for supporting datasets (population, dwellings/income, charger info, coordinates, stations) and creation of a master suburb table.

- `checkclustersb.ipynb`  
  Clustering experiments and cluster interpretation (k selection, cluster counts, cluster summary).

- `pricingmodel.ipynb`  
  Indicative pricing signal generation based on suburb/cluster characteristics (not a production-ready pricing model).

---

## 3) Datasets (Raw vs Clean)
Raw datasets are located in:
- `../datasets/`

Cleaned datasets are saved to:
- `../datasets/clean/`

Key cleaned outputs include:
- `suburb_population_clean.csv`
- `info_for_pcz_clean.csv`
- `stations_per_town_clean.csv`
- `charger_info_mel_clean.csv`
- `coordinates_clean.csv`
- `master_suburb_table.csv` (and variants with additional features)
- `clustered_suburbs_final.csv`
- `optimal_pricing_by_suburb.csv`

---

## 4) High-level Workflow (Repro Steps)
1. Run `01_data_review_cleaning.ipynb` to:
   - standardise suburb naming
   - clean numeric fields (income, dwellings, etc.)
   - create `master_suburb_table*.csv`

2. Run clustering notebook (`checkclustersb.ipynb`) to:
   - scale features
   - select k (elbow method)
   - assign cluster IDs and cluster labels
   - export clustered results

3. Run pricing notebook (`pricingmodel.ipynb`) to:
   - generate indicative pricing signals by suburb/cluster
   - export `optimal_pricing_by_suburb.csv`

---

## 5) Notes / Assumptions
- Some datasets do not fully overlap by suburb naming conventions; suburb keys were standardised to maximise join coverage.
- Congestion values were imputed using a global median when unmatched to suburb-level keys (for completeness in clustering features).
- Latitude/longitude coverage is limited for some suburbs; location enrichment is a future improvement.

---

## 6) Proposed API Output (Decision-support)
Recommended minimal API payload:
- `suburb`
- `cluster_id`
- `cluster_label`
- `summary_metrics` (population, chargers, stations, ratios)
- `pricing_signal` (indicative price)

See `../api/openapi.yaml` for the draft schema.
