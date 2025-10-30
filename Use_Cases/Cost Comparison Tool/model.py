# improved_forecast_dummy.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# -----------------------------
# 1) Load data
# -----------------------------
DATA_PATH = "dummy_data.csv"   # change if needed
df = pd.read_csv(DATA_PATH)

# -----------------------------
# 2) Basic sanity / outlier handling
#    (light, conservative clipping – adjust if you like)
# -----------------------------
# Clip impossible/obvious mistakes:
df["petrol_price_per_l"] = df["petrol_price_per_l"].clip(lower=0, upper=5.0)
df["electricity_price_per_kwh"] = df["electricity_price_per_kwh"].clip(lower=0, upper=2.0)
df["distance_km"] = df["distance_km"].clip(lower=0, upper=df["distance_km"].quantile(0.99))
df["ice_eff_l_per_100km"] = df["ice_eff_l_per_100km"].clip(lower=0)

# Optional robust winsorization for target to reduce influence of extreme points
y_raw = df["savings_ice_minus_ev"].copy()
q_low, q_high = y_raw.quantile([0.01, 0.99])
y = y_raw.clip(q_low, q_high)

# -----------------------------
# 3) Feature engineering
# -----------------------------
# Assumption: average EV consumption ~ 0.15 kWh/km (tune per your fleet)
EV_KWH_PER_KM = 0.15

df["fuel_cost_per_km"] = (df["ice_eff_l_per_100km"] / 100.0) * df["petrol_price_per_l"]
df["ev_cost_per_km"] = df["electricity_price_per_kwh"] * EV_KWH_PER_KM
df["distance_x_petrol"] = df["distance_km"] * df["petrol_price_per_l"]
df["distance_x_elec"] = df["distance_km"] * df["electricity_price_per_kwh"]
df["eff_ratio"] = df["ice_eff_l_per_100km"] / (EV_KWH_PER_KM * 100)  # ICE L/100km vs EV kWh/100km (scaled)

base_features = [
    "distance_km",
    "electricity_price_per_kwh",
    "ice_eff_l_per_100km",
    "petrol_price_per_l",
    "fuel_cost_per_km",
    "ev_cost_per_km",
    "distance_x_petrol",
    "distance_x_elec",
    "eff_ratio",
]

X = df[base_features].copy()

# -----------------------------
# 4) Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------
# 5) Modeling pipelines
# -----------------------------
numeric_features = base_features
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

preprocess = ColumnTransformer(
    transformers=[("num", numeric_transformer, numeric_features)],
    remainder="drop",
)

models = {
    # Plain scaled linear regression
    "LinearRegression": Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("reg", LinearRegression())
        ]
    ),
    # Ridge can be a bit more stable than OLS
    "Ridge(alpha=1.0)": Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("reg", Ridge(alpha=1.0, random_state=42))
        ]
    ),
    # Polynomial (degree=2) + Ridge to avoid overfitting
    "Poly2 + Ridge": Pipeline(
        steps=[
            ("poly_prep",
             ColumnTransformer(
                 transformers=[
                     ("poly",
                      Pipeline([
                          ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                          ("scaler", StandardScaler())
                      ]),
                      numeric_features),
                 ],
                 remainder="drop",
             )),
            ("reg", Ridge(alpha=1.0, random_state=42)),
        ]
    ),
    # Tree-based (handles non-linearity & interactions)
    "RandomForest": Pipeline(
        steps=[
            ("preprocess", "passthrough"),  # trees don't need scaling
            ("reg", RandomForestRegressor(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ]
    ),
    "GradientBoosting": Pipeline(
        steps=[
            ("preprocess", "passthrough"),
            ("reg", GradientBoostingRegressor(
                n_estimators=350,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                random_state=42
            ))
        ]
    ),
}

# -----------------------------
# 6) Train, evaluate, compare
# -----------------------------
def evaluate(y_true, y_pred, label):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return {"Model": label, "R2": r2, "RMSE": rmse, "MAE": mae}

results = []
trained = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    results.append(evaluate(y_test, y_pred, name))
    trained[name] = pipe

score_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\n=== Model Scorecard (higher R2 is better; lower RMSE/MAE is better) ===")
print(score_df.to_string(index=False))

# -----------------------------
# 7) Best model diagnostics
# -----------------------------
best_name = score_df.iloc[0]["Model"]
best_model = trained[best_name]
print(f"\nBest model: {best_name}")

# Parity plot for the best model
y_pred_best = best_model.predict(X_test)
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_best, alpha=0.7)
mn, mx = y_test.min(), y_test.max()
plt.plot([mn, mx], [mn, mx], "r--")
plt.xlabel("Actual Savings ($)")
plt.ylabel("Predicted Savings ($)")
plt.title(f"Actual vs Predicted — {best_name}")
plt.tight_layout()
plt.show()

# -----------------------------
# 8) Feature importance (for tree models)
# -----------------------------
def plot_importance(model, feature_names, title):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importances)), importances[order])
    plt.xticks(range(len(importances)), np.array(feature_names)[order], rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()

if "RandomForest" in best_name:
    rf = best_model.named_steps["reg"]
    plot_importance(rf, base_features, "Random Forest Feature Importance")
elif "GradientBoosting" in best_name:
    gb = best_model.named_steps["reg"]
    plot_importance(gb, base_features, "Gradient Boosting Feature Importance")

# -----------------------------
# 9) Simple scenario forecasting helper
# -----------------------------
def forecast_savings(
    distance_km,
    electricity_price_per_kwh,
    ice_eff_l_per_100km,
    petrol_price_per_l,
    model=best_model
):
    row = pd.DataFrame([{
        "distance_km": distance_km,
        "electricity_price_per_kwh": electricity_price_per_kwh,
        "ice_eff_l_per_100km": ice_eff_l_per_100km,
        "petrol_price_per_l": petrol_price_per_l,
        "fuel_cost_per_km": (ice_eff_l_per_100km / 100.0) * petrol_price_per_l,
        "ev_cost_per_km": electricity_price_per_kwh * EV_KWH_PER_KM,
        "distance_x_petrol": distance_km * petrol_price_per_l,
        "distance_x_elec": distance_km * electricity_price_per_kwh,
        "eff_ratio": ice_eff_l_per_100km / (EV_KWH_PER_KM * 100),
    }])
    return float(model.predict(row)[0])

# Example forecast (tweak as needed)
example = forecast_savings(
    distance_km=250,
    electricity_price_per_kwh=0.30,
    ice_eff_l_per_100km=7.5,
    petrol_price_per_l=2.10
)
print(f"\nExample scenario forecasted savings: ${example:,.2f}")

# -----------------------------
# 10) Scenario forecasting (5–10 years horizon)
# -----------------------------

import matplotlib.pyplot as plt

def scenario_forecast(
    base_petrol=2.0,
    base_electricity=0.30,
    distance=300,
    ice_eff=7.5,
    years=10,
    petrol_growth=0.05,       # +5% per year
    electricity_growth=0.02   # +2% per year
):
    """Simulate yearly petrol/electricity price growth and forecast savings."""
    forecasts = []
    for yr in range(1, years + 1):
        petrol_price = base_petrol * (1 + petrol_growth) ** yr
        elec_price = base_electricity * (1 + electricity_growth) ** yr
        savings = forecast_savings(
            distance_km=distance,
            electricity_price_per_kwh=elec_price,
            ice_eff_l_per_100km=ice_eff,
            petrol_price_per_l=petrol_price,
            model=best_model
        )
        forecasts.append({
            "Year": 2025 + yr,
            "PetrolPrice": petrol_price,
            "ElecPrice": elec_price,
            "ForecastedSavings": savings
        })
    return pd.DataFrame(forecasts)

# Run 10-year scenario
scenario_df = scenario_forecast(years=10)

print("\n=== 10-Year Scenario Forecast ===")
print(scenario_df)

# Plot savings trend
plt.figure(figsize=(8, 5))
plt.plot(scenario_df["Year"], scenario_df["ForecastedSavings"], marker="o", color="green")
plt.xlabel("Year")
plt.ylabel("Forecasted Savings ($)")
plt.title("Forecasted EV vs ICE Savings (10-Year Scenario)")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 10b) Multi-scenario forecasting & comparison plot
# -----------------------------

def scenario_forecast(
    base_petrol=2.0,
    base_electricity=0.30,
    distance=300,
    ice_eff=7.5,
    years=10,
    petrol_growth=0.05,       # annual petrol growth
    electricity_growth=0.02   # annual electricity growth
):
    rows = []
    start_year = 2025  # set as needed
    for offset in range(1, years + 1):
        yr = start_year + offset
        petrol_price = base_petrol * (1 + petrol_growth) ** offset
        elec_price = base_electricity * (1 + electricity_growth) ** offset
        savings = forecast_savings(
            distance_km=distance,
            electricity_price_per_kwh=elec_price,
            ice_eff_l_per_100km=ice_eff,
            petrol_price_per_l=petrol_price,
            model=best_model
        )
        rows.append({
            "Year": yr,
            "PetrolPrice": petrol_price,
            "ElecPrice": elec_price,
            "ForecastedSavings": savings
        })
    return pd.DataFrame(rows)

# Define scenarios: (petrol_growth, electricity_growth)
scenarios = {
    "Low (2% petrol, 1% elec)":  (0.02, 0.01),
    "Med (5% petrol, 2% elec)":  (0.05, 0.02),
    "High (8% petrol, 3.5% elec)": (0.08, 0.035),
}

# Run scenarios
scenario_frames = []
for name, (pg, eg) in scenarios.items():
    df_s = scenario_forecast(
        base_petrol=2.0,
        base_electricity=0.30,
        distance=300,          # adjust if you want a different representative trip
        ice_eff=7.5,
        years=10,
        petrol_growth=pg,
        electricity_growth=eg
    )
    df_s["Scenario"] = name
    scenario_frames.append(df_s)

all_scenarios = pd.concat(scenario_frames, ignore_index=True)

# Optional: inspect table
print("\n=== Multi-Scenario 10-Year Forecasts ===")
print(all_scenarios.pivot(index="Year", columns="Scenario", values="ForecastedSavings").round(2))

# Plot all scenarios on one chart
plt.figure(figsize=(9, 5.5))
for name, _ in scenarios.items():
    df_plot = all_scenarios[all_scenarios["Scenario"] == name]
    plt.plot(df_plot["Year"], df_plot["ForecastedSavings"], marker="o", label=name)

plt.xlabel("Year")
plt.ylabel("Forecasted Savings ($)")
plt.title("EV vs ICE Savings — Multi-Scenario 10-Year Forecast")
plt.grid(True)
plt.legend(title="Scenario")
plt.tight_layout()
plt.show()
