import os
import re
import math
import json
import argparse
from typing import List, Optional, Tuple, Dict

import pandas as pd

try:
    # Optional: only needed if user chooses "auto fetch price"
    import requests
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    requests = None
    BeautifulSoup = None


EV_CSV = "test.ev_vehicles.csv"
STATIONS_CSV = "test.charging_stations.csv"

AU_STATES = [
    "NSW", "VIC", "QLD", "SA", "WA", "TAS", "ACT", "NT"
]
STATE_NAMES = {
    "NSW": "New South Wales",
    "VIC": "Victoria",
    "QLD": "Queensland",
    "SA": "South Australia",
    "WA": "Western Australia",
    "TAS": "Tasmania",
    "ACT": "Australian Capital Territory",
    "NT": "Northern Territory",
}


# -----------------------------
# I/O helpers
# -----------------------------
def prompt_int_in_range(prompt: str, lo: int, hi: int) -> int:
    while True:
        try:
            v = int(input(prompt).strip())
            if lo <= v <= hi:
                return v
        except Exception:
            pass
        print(f"Please enter a number between {lo} and {hi}.")


def prompt_float(prompt: str, default: Optional[float] = None, min_val: Optional[float] = None) -> float:
    while True:
        s = input(f"{prompt}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if s == "" and default is not None:
            return default
        try:
            v = float(s)
            if min_val is not None and v < min_val:
                print(f"Value must be >= {min_val}.")
                continue
            return v
        except Exception:
            print("Please enter a valid number.")


def prompt_choice(prompt: str, options: List[str]) -> int:
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    return prompt_int_in_range(prompt, 1, len(options))


# -----------------------------
# Dataset loading & utilities
# -----------------------------
def load_csv_safely(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    return df


def first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def available_values(df: pd.DataFrame, column_candidates: List[str]) -> Tuple[str, List[str]]:
    col = first_col(df, column_candidates)
    if not col:
        raise ValueError(f"None of these columns found: {column_candidates}")
    vals = (
        df[col].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()
    )
    vals_sorted = sorted(vals, key=lambda x: x.lower())
    return col, vals_sorted


def filter_df(df: pd.DataFrame, col: str, value: str) -> pd.DataFrame:
    return df[df[col].astype(str).str.strip().str.lower() == value.strip().lower()]


# ----------------------------------------
# EV efficiency inference (kWh per km)
# ----------------------------------------
def infer_ev_efficiency_kwh_per_km(row: pd.Series) -> Optional[float]:
    # direct
    for col in ["efficiency_kwh_per_km", "kwh_per_km", "efficiency_kwhkm", "ev_efficiency"]:
        if col in row and pd.notna(row[col]):
            try:
                val = float(row[col])
                if val > 0:
                    return val
            except Exception:
                pass
    # per 100km
    for col in ["efficiency_kwh_per_100km", "kwh_per_100km"]:
        if col in row and pd.notna(row[col]):
            try:
                val = float(row[col])
                if val > 0:
                    return val / 100.0
            except Exception:
                pass
    # derive from battery & range
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
    return None


def ev_identity(row: pd.Series) -> str:
    pieces = []
    for cand in ["make", "brand", "manufacturer"]:
        if cand in row and pd.notna(row[cand]):
            pieces.append(str(row[cand]))
            break
    for cand in ["model", "name"]:
        if cand in row and pd.notna(row[cand]):
            pieces.append(str(row[cand]))
            break
    for cand in ["variant", "trim"]:
        if cand in row and pd.notna(row[cand]):
            pieces.append(str(row[cand]))
            break
    return " ".join(pieces) if pieces else "Unknown EV"


# ----------------------------------------
# Electricity price fetchers (AU)
# ----------------------------------------
FINDER_URL = "https://www.finder.com.au/energy/electricity/average-cost-of-electricity"
CANSTAR_URL = "https://www.canstarblue.com.au/electricity/electricity-costs-kwh/"

STATE_PATTERNS = {
    # Attach multiple name variants for robustness
    "NSW": [r"New South Wales", r"\bNSW\b"],
    "VIC": [r"Victoria", r"\bVIC\b"],
    "QLD": [r"Queensland", r"\bQLD\b"],
    "SA": [r"South Australia", r"\bSA\b(?!\w)"],
    "WA": [r"Western Australia", r"\bWA\b(?!\w)"],
    "TAS": [r"Tasmania", r"\bTAS\b"],
    "ACT": [r"Australian Capital Territory", r"\bACT\b"],
    "NT": [r"Northern Territory", r"\bNT\b"],
}


def parse_prices_from_html(html: str) -> Dict[str, float]:
    """
    Pulls first 'c/kWh' number near each state name.
    Returns { 'NSW': 0.33, ... } in $/kWh.
    """
    prices: Dict[str, float] = {}
    # collapse whitespace
    text = re.sub(r"\s+", " ", html)
    # Search in a window around state names
    for st, patterns in STATE_PATTERNS.items():
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                start = max(0, m.start() - 200)
                end = min(len(text), m.end() + 200)
                window = text[start:end]
                # find like '32.5c/kWh' or '0.32 $/kWh'
                m2 = re.search(r"(\d{1,2}\.?\d{0,2})\s*c\s*/\s*kWh", window, flags=re.IGNORECASE)
                if not m2:
                    m2 = re.search(r"\$?\s*(0\.\d{2})\s*/\s*kWh", window, flags=re.IGNORECASE)
                    if m2:
                        val = float(m2.group(1))  # already $/kWh
                        prices[st] = val
                        break
                else:
                    cents = float(m2.group(1))
                    prices[st] = cents / 100.0  # convert to $/kWh
                    break
            if st in prices:
                break
    return prices


def fetch_au_prices() -> Dict[str, float]:
    """Try Finder then Canstar. Return map of state -> $/kWh"""
    out: Dict[str, float] = {}
    if requests is None:
        return out
    headers = {"User-Agent": "Mozilla/5.0 (compatible; EVTool/1.0)"}
    for url in [FINDER_URL, CANSTAR_URL]:
        try:
            resp = requests.get(url, timeout=8, headers=headers)
            if resp.ok:
                prices = parse_prices_from_html(resp.text)
                out.update(prices)
            if len(out) >= 4:  # got a few states; good enough
                break
        except Exception:
            continue
    return out


# ----------------------------------------
# Cost math
# ----------------------------------------
def ev_trip_cost(distance_km: float, ev_eff_kwh_per_km: float, elec_price_per_kwh: float) -> float:
    return distance_km * ev_eff_kwh_per_km * elec_price_per_kwh


def ice_trip_cost(distance_km: float, l_per_100km: float, fuel_price_per_l: float) -> float:
    return (l_per_100km / 100.0) * distance_km * fuel_price_per_l


# ----------------------------------------
# Interactive flow
# ----------------------------------------
def interactive_flow():
    print("=== EV vs ICE Trip Cost (Interactive) ===\n")

    ev_df = load_csv_safely(EV_CSV)

    # Choose Make
    make_col, makes = available_values(ev_df, ["make", "brand", "manufacturer"])
    # Bubble up common brands first if present
    priority = ["BYD", "Tesla", "BMW"]
    makes_sorted = sorted(makes, key=lambda x: (0 if x in priority else 1, x.lower()))
    print("Select a Make:")
    idx = prompt_choice("Enter number: ", makes_sorted)
    chosen_make = makes_sorted[idx - 1]

    # Choose Model
    subset_model = filter_df(ev_df, make_col, chosen_make)
    model_col, models = available_values(subset_model, ["model", "name"])
    print(f"\nSelect a Model for {chosen_make}:")
    idx = prompt_choice("Enter number: ", models)
    chosen_model = models[idx - 1]

    # Choose Variant (optional)
    subset_variant = subset_model[subset_model[model_col].astype(str).str.lower() == chosen_model.lower()]
    variant_col = first_col(subset_variant, ["variant", "trim"])
    chosen_variant = None
    if variant_col:
        variants = (
            subset_variant[variant_col]
            .dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()
        )
        if variants:
            variants_sorted = sorted(variants, key=lambda x: x.lower())
            print(f"\nSelect a Variant for {chosen_model} (or 0 to skip):")
            for i, v in enumerate(variants_sorted, 1):
                print(f"{i}. {v}")
            print("0. Skip")
            sel = prompt_int_in_range("Enter number: ", 0, len(variants_sorted))
            if sel != 0:
                chosen_variant = variants_sorted[sel - 1]
                subset_variant = subset_variant[subset_variant[variant_col].astype(str).str.lower() == chosen_variant.lower()]

    # Select the first matching row as the EV choice
    chosen_ev = subset_variant.iloc[0] if not subset_variant.empty else subset_model[subset_model[model_col].astype(str).str.lower() == chosen_model.lower()].iloc[0]

    ev_name = ev_identity(chosen_ev)
    ev_eff = infer_ev_efficiency_kwh_per_km(chosen_ev) or 0.15
    print(f"\nSelected EV: {ev_name}")
    print(f"Inferred efficiency: {ev_eff:.3f} kWh/km")

    # Trip distance
    distance_km = prompt_float("\nEnter trip distance (km)", min_val=0.1)

    # Electricity price: auto or manual
    print("\nElectricity price source:")
    src_choice = prompt_choice("1=Auto (internet)  2=Manual  3=Use average of stations file\nEnter number: ", ["Auto (internet)", "Manual", "Average of stations file"])

    elec_price = 0.30
    price_note = "default"
    if src_choice == 1:
        # Ask for state (makes output more meaningful)
        print("\nChoose your State/Territory:")
        disp = [f"{k} - {STATE_NAMES[k]}" for k in AU_STATES]
        sidx = prompt_choice("Enter number: ", disp)
        state = AU_STATES[sidx - 1]

        prices = fetch_au_prices()
        if state in prices:
            elec_price = prices[state]
            price_note = f"Auto (Finder/Canstar scrape) for {state}"
        elif prices:
            # fallback to any price we got (mean)
            elec_price = sum(prices.values()) / len(prices)
            price_note = f"Auto (mean of {len(prices)} states from Finder/Canstar)"
        else:
            print("[WARN] Could not fetch from internet (or libraries missing). Falling back to $0.30/kWh.")
            elec_price = 0.30
            price_note = "fallback default"
    elif src_choice == 2:
        elec_price = prompt_float("Enter electricity price ($/kWh), e.g., 0.33", min_val=0.05)
        price_note = "manual"
    else:
        # mean of stations file if present
        try:
            st_df = load_csv_safely(STATIONS_CSV)
            def station_price(row):
                for col in ["price_per_kwh", "electricity_price", "price", "kwh_price"]:
                    if col in row and pd.notna(row[col]):
                        try:
                            val = float(row[col])
                            if val > 0:
                                return val
                        except Exception:
                            pass
                return None
            prices = st_df.apply(station_price, axis=1).dropna().tolist()
            if prices:
                elec_price = float(sum(prices) / len(prices))
                price_note = f"average of {len(prices)} stations"
            else:
                price_note = "fallback default"
        except Exception:
            price_note = "fallback default"

    # ICE benchmarks
    ice_l_per_100 = prompt_float("\nEnter ICE fuel efficiency (L/100km)", default=7.0, min_val=1.0)
    ice_fuel_price = prompt_float("Enter petrol price ($/L)", default=2.0, min_val=0.5)

    # Compute
    ev_cost = distance_km * ev_eff * elec_price
    ice_cost = (ice_l_per_100 / 100.0) * distance_km * ice_fuel_price
    savings = ice_cost - ev_cost

    # Emissions (rough)
    grid_emission_factor = 0.70  # kg CO2 per kWh
    petrol_emission_factor = 2.31  # kg CO2 per L
    ev_co2 = distance_km * ev_eff * grid_emission_factor
    ice_co2 = (ice_l_per_100 / 100.0) * distance_km * petrol_emission_factor

    print("\n=== Results ===")
    print(f"Trip distance:          {distance_km:.1f} km")
    print(f"EV selected:            {ev_name}")
    print(f"EV efficiency:          {ev_eff:.3f} kWh/km")
    print(f"Electricity price:      ${elec_price:.3f}/kWh  [{price_note}]")
    print(f"ICE efficiency:         {ice_l_per_100:.2f} L/100km")
    print(f"ICE fuel price:         ${ice_fuel_price:.2f}/L")
    print("\n--- Costs ---")
    print(f"EV trip cost:           ${ev_cost:.2f}")
    print(f"ICE trip cost:          ${ice_cost:.2f}")
    print(f"Savings (ICE - EV):     ${savings:.2f}")
    print("\n--- Emissions (approx.) ---")
    print(f"EV CO2:                 {ev_co2:.2f} kg")
    print(f"ICE CO2:                {ice_co2:.2f} kg")
    print(f"CO2 saved:              {ice_co2 - ev_co2:.2f} kg")
    print("\nDone.")


# ----------------------------------------
# CLI wrapper: supports --interactive flag
# ----------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EV vs ICE Cost Tool (interactive or CLI).")
    parser.add_argument("--interactive", action="store_true", help="Run interactive prompts.")
    args, _ = parser.parse_known_args()

    if args.interactive:
        interactive_flow()
    else:
        print("Run with --interactive for the guided experience.")
        print("Example: python app.py --interactive")


if __name__ == "__main__":
    main()
