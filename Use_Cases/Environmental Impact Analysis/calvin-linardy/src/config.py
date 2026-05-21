"""
config.py
---------
Central constants for the Environmental Impact Analysis model.
All emission factors are sourced from DCCEEW National Greenhouse Accounts
Factors 2023 (most recent published edition).
"""

# ---------------------------------------------------------------------------
# Grid emission factors (kg CO2-e per kWh delivered at the meter)
# Source: DCCEEW National Greenhouse Accounts Factors 2023, Table 3
# These are Scope 2 location-based factors for grid electricity.
# ---------------------------------------------------------------------------
GRID_EMISSION_FACTORS = {
    "NSW": 0.790,
    "ACT": 0.000,   # 100 % renewable electricity purchase agreements
    "VIC": 0.990,
    "QLD": 0.810,
    "SA":  0.290,
    "WA":  0.650,   # South-West Interconnected System (SWIS)
    "TAS": 0.130,
    "NT":  0.590,
}

# ---------------------------------------------------------------------------
# Fuel emission factors (kg CO2-e per litre of fuel, Scope 1 combustion)
# Source: DCCEEW National Greenhouse Accounts Factors 2023, Table 1
# ---------------------------------------------------------------------------
FUEL_EMISSION_FACTORS = {
    "Petrol":   2.289,
    "Petrol95": 2.289,
    "Petrol98": 2.289,
    "E10":      2.195,
    "Diesel":   2.703,
    "LPG":      1.542,
}

# Aliases accepted by the API (maps user-supplied strings to canonical keys)
FUEL_TYPE_ALIASES = {
    "petrol":    "Petrol",
    "petrol91":  "Petrol",
    "petrol 91": "Petrol",
    "petrol95":  "Petrol95",
    "petrol 95": "Petrol95",
    "petrol98":  "Petrol98",
    "petrol 98": "Petrol98",
    "e10":       "E10",
    "diesel":    "Diesel",
    "lpg":       "LPG",
}

# ---------------------------------------------------------------------------
# Lifecycle / manufacturing constants
# ---------------------------------------------------------------------------

# Battery manufacturing emission intensity (kg CO2-e per kWh of pack capacity)
# Source: ICCT 2021 "Life-cycle greenhouse gas emissions of current light-duty
# vehicles in major vehicle markets", median global estimate.
BATTERY_MANUFACTURING_KG_CO2_PER_KWH = 65.0

# EV assembly overhead above a comparable ICE vehicle (kg CO2-e per vehicle)
# Excludes battery; covers aluminium body, electric motor, inverter, etc.
# Source: Transport & Environment 2021 lifecycle study, EU average.
EV_ASSEMBLY_OVERHEAD_KG_CO2 = 1_500.0

# Typical end-of-life battery second-life credit (kg CO2-e avoided, per vehicle)
# Accounts for second-life stationary storage or responsible recycling.
BATTERY_EOL_CREDIT_KG_CO2 = 200.0

# ---------------------------------------------------------------------------
# Default usage assumptions
# Source: ABS Motor Vehicle Census 2020 (most recent)
# ---------------------------------------------------------------------------
DEFAULT_ANNUAL_KM = 13_100          # km/year, Australian average
DEFAULT_VEHICLE_LIFETIME_YEARS = 15  # years
DEFAULT_LIFETIME_KM = DEFAULT_ANNUAL_KM * DEFAULT_VEHICLE_LIFETIME_YEARS

# Electricity price reference (AUD/kWh) — used for cost comparison only
# Source: Australian Energy Market Commission, residential average 2024
ELECTRICITY_PRICE_AUD_PER_KWH = 0.30

# Petrol/diesel reference prices (AUD/L) — 2024 Australian annual average
FUEL_PRICES_AUD_PER_L = {
    "Petrol":   2.00,
    "Petrol95": 2.05,
    "Petrol98": 2.15,
    "E10":      1.95,
    "Diesel":   2.10,
    "LPG":      0.85,
}

# ---------------------------------------------------------------------------
# Real-world adjustment bounds
# These clamp XGBoost predictions to physically plausible values.
# ---------------------------------------------------------------------------
RW_ADJUSTMENT_MIN = 0.80   # Best-case: ideal conditions (mild, city driving, new battery)
RW_ADJUSTMENT_MAX = 1.50   # Worst-case: extreme cold, highway, aged battery
