"""
train.py
--------
CLI entry point for training and saving the XGBoost real-world adjustment model.

Usage
-----
    python train.py                     # default 8000 samples
    python train.py --samples 15000     # larger synthetic dataset
    python train.py --seed 123          # reproducibility
    python train.py --evaluate          # run evaluation examples after training

The trained model is saved to models/rw_adjustment_xgb.pkl.
"""

import argparse
import sys
from pathlib import Path

# Ensure the package root is on the path regardless of where the script is run from
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import model as xgb_model
from src import calculator
from src.config import DEFAULT_ANNUAL_KM


def run_evaluation_examples(model_bundle: dict):
    """Print a set of sanity-check predictions to verify the trained model."""
    print("\n" + "=" * 60)
    print("EVALUATION EXAMPLES")
    print("=" * 60)

    examples = [
        {
            "label": "Tesla Model 3 | NSW | Mixed | New",
            "ev_kwh": 14.9,
            "ev_battery": 57.5,
            "ice_l": 6.0,
            "ice_fuel": "Petrol",
            "state": "NSW",
            "temp": 17.7,
            "style": "mixed",
            "age": 0,
        },
        {
            "label": "Tesla Model 3 | SA (renewables) | Mixed | New",
            "ev_kwh": 14.9,
            "ev_battery": 57.5,
            "ice_l": 6.0,
            "ice_fuel": "Petrol",
            "state": "SA",
            "temp": 17.3,
            "style": "mixed",
            "age": 0,
        },
        {
            "label": "BYD Atto 3 | VIC (coal) | Highway | 5yr old",
            "ev_kwh": 17.5,
            "ev_battery": 60.5,
            "ice_l": 8.2,
            "ice_fuel": "Diesel",
            "state": "VIC",
            "temp": 14.9,
            "style": "highway",
            "age": 5,
        },
        {
            "label": "Kia EV9 | QLD (tropical) | City | New",
            "ev_kwh": 23.9,
            "ev_battery": 99.8,
            "ice_l": 8.8,
            "ice_fuel": "Petrol",
            "state": "QLD",
            "temp": 25.0,
            "style": "city",
            "age": 0,
        },
        {
            "label": "MG4 | TAS (hydro) | Mixed | 3yr old",
            "ev_kwh": 15.3,
            "ev_battery": 51.0,
            "ice_l": 7.2,
            "ice_fuel": "Diesel",
            "state": "TAS",
            "temp": 12.4,
            "style": "mixed",
            "age": 3,
        },
    ]

    for ex in examples:
        rw = xgb_model.predict_adjustment(
            avg_temp_celsius=ex["temp"],
            driving_style=ex["style"],
            vehicle_age_years=ex["age"],
            battery_capacity_kwh=ex["ev_battery"],
            model_bundle=model_bundle,
        )
        result = calculator.calculate(
            ev_consumption_kwh_per_100km=ex["ev_kwh"],
            ev_battery_kwh=ex["ev_battery"],
            ice_consumption_l_per_100km=ex["ice_l"],
            ice_fuel_type=ex["ice_fuel"],
            state=ex["state"],
            real_world_adjustment=rw,
            annual_km=DEFAULT_ANNUAL_KM,
            vehicle_lifetime_years=15,
            driving_style=ex["style"],
            include_lifecycle=True,
        )

        print(f"\n  {ex['label']}")
        print(f"    RW adjustment factor : {rw:.4f}")
        print(f"    EV  total  (g CO2/km): {result.ev_total_g_per_km:.1f}  "
              f"(operational: {result.ev_operational_g_per_km:.1f}, "
              f"manufacturing: {result.ev_manufacturing_g_per_km:.1f})")
        print(f"    ICE total  (g CO2/km): {result.ice_operational_g_per_km:.1f}")
        print(f"    CO2 saving (g CO2/km): {result.co2_savings_g_per_km:.1f}  "
              f"({result.percentage_reduction:.1f}% reduction)")
        print(f"    Annual saving         : {result.co2_savings_kg_per_year:.0f} kg CO2/yr  "
              f"≈ {result.trees_planted_equivalent_per_year} trees/yr")
        print(f"    Lifetime saving       : {result.co2_savings_tonnes_lifetime:.1f} t CO2")
        print(f"    Cost saving/year      : AUD ${result.cost_saving_aud_per_year:,.0f}")


def main():
    parser = argparse.ArgumentParser(
        description="Train the EVAT Environmental Impact XGBoost model."
    )
    parser.add_argument(
        "--samples", type=int, default=8_000,
        help="Number of synthetic training samples to generate (default: 8000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run evaluation examples after training"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("EVAT Environmental Impact Analysis — Model Training")
    print("=" * 60)

    metrics = xgb_model.train(n_samples=args.samples, seed=args.seed)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Samples  : {metrics['n_train']:,} train / {metrics['n_test']:,} test")
    print(f"  MAE      : {metrics['mae']:.6f} (mean adjustment factor error)")
    print(f"  RMSE     : {metrics['rmse']:.6f}")
    print(f"  R²       : {metrics['r2']:.4f}")
    print(f"  CV R²    : {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
    print(f"\n  Feature importances:")
    for feat, imp in sorted(metrics["feature_importances"].items(), key=lambda x: -x[1]):
        print(f"    {feat:<30}: {imp:.4f}")

    if args.evaluate:
        bundle = xgb_model.load_model()
        run_evaluation_examples(bundle)

    print("\nDone. Start the API with:")
    print("  uvicorn api.main:app --reload --port 8001")


if __name__ == "__main__":
    main()
