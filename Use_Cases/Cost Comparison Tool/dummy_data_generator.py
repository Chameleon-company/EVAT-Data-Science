# %%writefile /mnt/data/dummy_simulator.py
import argparse
import random
from typing import Optional, Tuple, List, Dict

import pandas as pd


EV_CSV = "test.ev_vehicles.csv"
STATES = [
    ("NSW", "New South Wales"),
    ("VIC", "Victoria"),
    ("QLD", "Queensland"),
    ("SA", "South Australia"),
    ("WA", "Western Australia"),
    ("TAS", "Tasmania"),
    ("ACT", "Australian Capital Territory"),
    ("NT", "Northern Territory"),
]


# ---------------- Helpers ----------------
def first_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    m = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in m:
            return m[n.lower()]
    return None


def choose_make_model_variant(ev: pd.DataFrame) -> Tuple[str, str, str]:
    make_col = first_col(ev, ["make", "brand", "manufacturer"])
    model_col = first_col(ev, ["model", "name", "vehicle", "ev_model", "make_model"])
    variant_col = first_col(ev, ["variant", "trim"])

    if make_col is None or model_col is None:
        raise ValueError("Could not find make/model columns in EV CSV.")

    makes = sorted(ev[make_col].dropna().astype(str).unique().tolist(), key=str.lower)
    make = random.choice(makes)

    models = sorted(
        ev.loc[ev[make_col].astype(str).str.lower() == make.lower(), model_col]
        .dropna().astype(str).unique().tolist(),
        key=str.lower
    )
    model = random.choice(models) if models else ""

    variant = ""
    if variant_col and model:
        variants = sorted(
            ev.loc[
                (ev[make_col].astype(str).str.lower() == make.lower()) &
                (ev[model_col].astype(str).str.lower() == model.lower()),
            variant_col].dropna().astype(str).unique().tolist(),
            key=str.lower
        )
        if variants:
            variant = random.choice(variants)

    return make, model, variant


def ev_identity(make: str, model: str, variant: str) -> str:
    parts = [make, model, variant]
    return " ".join([p for p in parts if p])


def infer_ev_efficiency_kwh_per_km(row: pd.Series) -> float:
    for col in ["efficiency_kwh_per_km", "kwh_per_km", "efficiency_kwhkm", "ev_efficiency"]:
        if col in row and pd.notna(row[col]):
            try:
                v = float(row[col])
                if v > 0:
                    return v
            except Exception:
                pass
    for col in ["efficiency_kwh_per_100km", "kwh_per_100km"]:
        if col in row and pd.notna(row[col]):
            try:
                v = float(row[col])
                if v > 0:
                    return v / 100.0
            except Exception:
                pass
    # derive battery/range
    batt_col = None
    range_col = None
    for cand in ["battery_kwh", "battery_capacity_kwh", "battery_capacity"]:
        if cand in row and pd.notna(row[cand]):
            batt_col = cand
            break
    for cand in ["range_km", "wltp_range_km", "epa_range_km", "range"]:
        if cand in row and pd.notna(row[cand]):
            range_col = cand
            break
    if batt_col and range_col:
        try:
            batt = float(row[batt_col])
            rng = float(row[range_col])
            if batt > 0 and rng > 0:
                return batt / rng
        except Exception:
            pass
    return 0.15  # default


def maybe_extreme(p: float) -> bool:
    return random.random() < p


def triangular(a: float, b: float, c: float) -> float:
    return random.triangular(a, b, c)


def sample_distance() -> int:
    # Random realistic distances (km)
    return random.choice([100, 150, 200, 250, 300, 400, 500])


def sample_elec_price(extreme_prob: float) -> float:
    # Typical AU retail ~0.22–0.45; allow extremes (0.07 / 0.60) for stress tests
    if maybe_extreme(extreme_prob):
        return round(random.choice([0.07, 0.10, 0.60]), 3)
    return round(triangular(0.22, 0.45, 0.32), 3)


def sample_petrol_price(extreme_prob: float) -> float:
    # Typical ~1.60–2.40; allow absurd 189.74 for stress tests
    if maybe_extreme(extreme_prob):
        return random.choice([89.99, 189.74, 0.01])
    return round(triangular(1.60, 2.40, 1.95), 2)


def sample_ice_eff(extreme_prob: float) -> float:
    # Typical 5.5–12.0 L/100km; allow absurd 999 / 2018 for stress tests
    if maybe_extreme(extreme_prob):
        return random.choice([2018.0, 999.0, 0.5, 50.0])
    return round(triangular(5.5, 12.0, 8.5), 2)


# -------------- Core simulation --------------
def simulate_one(ev: pd.DataFrame, extreme_prob: float, use_fixed_distance: bool = False) -> Tuple[str, Dict]:
    make, model, variant = choose_make_model_variant(ev)

    # Pick row close to chosen vehicle for EV efficiency inference
    make_col = first_col(ev, ["make", "brand", "manufacturer"])
    model_col = first_col(ev, ["model", "name", "vehicle", "ev_model", "make_model"])
    row = ev.iloc[0]
    mask = pd.Series([True] * len(ev))
    if make_col:
        mask &= ev[make_col].astype(str).str.lower() == make.lower()
    if model_col:
        mask &= ev[model_col].astype(str).str.lower() == model.lower()
    if mask.any():
        row = ev[mask].iloc[0]

    ev_eff = infer_ev_efficiency_kwh_per_km(row)
    distance = 200 if use_fixed_distance else sample_distance()

    # Electricity & state
    elec_price = sample_elec_price(extreme_prob)
    st_idx = random.randint(1, len(STATES))
    state = STATES[st_idx - 1][0]

    # ICE & petrol (+sometimes emulate fetch failure to show warning path)
    fetch_failed = maybe_extreme(0.25)
    if fetch_failed:
        ice_eff = sample_ice_eff(extreme_prob)
        warn_line = "[WARN] Could not fetch ICE efficiency online. Switching to manual entry."
        manual_prompt_val = f"{ice_eff:g} "
        ice_year_prompt = "2018 "
    else:
        ice_eff = sample_ice_eff(0.0)  # plausible number
        warn_line = None
        manual_prompt_val = " "
        ice_year_prompt = "2024 "

    petrol_price = sample_petrol_price(extreme_prob)
    petrol_state_idx = random.randint(1, len(STATES))
    petrol_state = STATES[petrol_state_idx - 1][0]

    # Costs & emissions
    ev_cost = distance * ev_eff * elec_price
    ice_cost = (ice_eff / 100.0) * distance * petrol_price
    savings = ice_cost - ev_cost
    grid_ef = 0.70
    petrol_ef = 2.31
    ev_co2 = distance * ev_eff * grid_ef
    ice_co2 = (ice_eff / 100.0) * distance * petrol_ef
    co2_saved = ice_co2 - ev_co2

    # Build transcript
    title = ev_identity(make, model, variant)
    lines = []
    lines.append("=== EV vs ICE Trip Cost (Interactive) ===\n")
    lines.append(f"Selected EV: {title}")
    lines.append(f"Inferred efficiency: {ev_eff:0.3f} kWh/km\n")
    lines.append(f"Trip distance:          {float(distance):.1f} km")
    lines.append(f"Electricity price:      ${elec_price:0.3f}/kWh  [Dummy generated]")
    lines.append(f"ICE efficiency:         {ice_eff:0.2f} L/100km")
    lines.append(f"Petrol price:           ${petrol_price:0.2f}/L\n")
    lines.append("--- Costs ---")
    lines.append(f"EV trip cost:           ${ev_cost:0.2f}")
    lines.append(f"ICE trip cost:          ${ice_cost:0.2f}")
    lines.append(f"Savings (ICE - EV):     ${savings:0.2f}\n")
    lines.append("--- Emissions (approx.) ---")
    lines.append(f"EV CO2:                 {ev_co2:0.2f} kg")
    lines.append(f"ICE CO2:                {ice_co2:0.2f} kg")
    lines.append(f"CO2 saved:              {co2_saved:0.2f} kg\n")
    lines.append("Done.")

    # Structured record for CSV
    rec = {
        "selected_ev": title,
        "distance_km": float(distance),
        "electricity_price_per_kwh": float(elec_price),
        "ice_eff_l_per_100km": float(ice_eff),
        "petrol_price_per_l": float(petrol_price),
        "ev_trip_cost": float(round(ev_cost, 2)),
        "ice_trip_cost": float(round(ice_cost, 2)),
        "savings_ice_minus_ev": float(round(savings, 2)),
        "ev_co2_kg": float(round(ev_co2, 2)),
        "ice_co2_kg": float(round(ice_co2, 2)),
        "co2_saved_kg": float(round(co2_saved, 2)),
        "state_elec": state,
        "state_petrol": petrol_state,
    }
    if warn_line:
        # include note when we simulated a fetch failure
        rec["notes"] = "ICE fetch failed; manual entry path simulated"
    return "\n".join(lines), rec


# -------------- CLI --------------
def main():
    ap = argparse.ArgumentParser(description="Generate dummy interactive transcripts and structured CSV for EV vs ICE tool.")
    ap.add_argument("--runs", type=int, default=5, help="How many transcripts/records to generate.")
    ap.add_argument("--extreme-prob", type=float, default=0.15, help="Probability of extreme dummy values (0–1).")
    ap.add_argument("--fixed-distance", action="store_true", help="Use fixed 200 km for all runs.")
    ap.add_argument("--out", default="/mnt/data/dummy_transcripts.txt", help="Output text file for transcripts.")
    ap.add_argument("--csv-out", default=None, help="Optional CSV path to save structured results.")
    ap.add_argument("--xlsx-out", default=None, help="Optional Excel path to save structured results.")
    args = ap.parse_args()

    ev = pd.read_csv(EV_CSV)

    transcripts: List[str] = []
    records: List[Dict] = []
    for _ in range(args.runs):
        t, rec = simulate_one(ev, extreme_prob=args.extreme_prob, use_fixed_distance=args.fixed_distance)
        transcripts.append(t)
        records.append(rec)

    # Write transcripts
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(transcripts))

    # Write CSV/XLSX if requested
    df = pd.DataFrame(records)
    if args.csv_out:
        df.to_csv(args.csv_out, index=False)
    if args.xlsx_out:
        with pd.ExcelWriter(args.xlsx_out, engine="xlsxwriter") as wr:
            df.to_excel(wr, sheet_name="DummyData", index=False)
            ws = wr.sheets["DummyData"]
            for i, col in enumerate(df.columns):
                width = max(12, min(40, int(df[col].astype(str).str.len().quantile(0.9)) + 3))
                ws.set_column(i, i, width)

    print(f"Wrote transcripts -> {args.out}")
    if args.csv_out:
        print(f"Wrote CSV -> {args.csv_out}")
    if args.xlsx_out:
        print(f"Wrote Excel -> {args.xlsx_out}")


if __name__ == "__main__":
    main()
