# %%writefile /mnt/data/app.py
import os
import re
import math
import json
import argparse
from typing import List, Optional, Tuple, Dict

import pandas as pd

# Optional: network libs for live price/efficiency fetch
try:
    import requests
except Exception:
    requests = None

EV_CSV = "test.ev_vehicles.csv"
STATIONS_CSV = "test.charging_stations.csv"
ICE_LOCAL_CSV = "ice_vehicles.csv"  # optional user-supplied: make, model, variant, year, l_per_100km

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
# Console helpers
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
# CSV & field helpers
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
    text = re.sub(r"\s+", " ", html)
    for st, patterns in STATE_PATTERNS.items():
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                start = max(0, m.start() - 250)
                end = min(len(text), m.end() + 250)
                window = text[start:end]
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


def fetch_au_elec_prices() -> Dict[str, float]:
    """Try Finder then Canstar for $/kWh per AU state."""
    out: Dict[str, float] = {}
    if requests is None:
        return out
    headers = {"User-Agent": "Mozilla/5.0 (compatible; EVTool/1.1)"}
    for url in [FINDER_URL, CANSTAR_URL]:
        try:
            resp = requests.get(url, timeout=8, headers=headers)
            if resp.ok:
                out.update(parse_prices_from_html(resp.text))
            if len(out) >= 4:
                break
        except Exception:
            continue
    return out


# ----------------------------------------
# Petrol price fetchers (AU)
# ----------------------------------------
NSW_FUELCHECK_PRODUCT = {
    "U91": 1, "E10": 2, "P95": 3, "P98": 4, "Diesel": 5, "LPG": 6
}

def fetch_petrol_price_au(state_code: Optional[str] = None, product: str = "U91") -> Optional[float]:
    """
    Returns $/L for given state (approx). Tries:
     1) NSW FuelCheck (NSW/TAS) avg price if API key present (env: FUELCHECK_API_KEY)
     2) WA FuelWatch average (U91) via public RSS JSON-ish endpoint
     3) GlobalPetrolPrices Australia page (AUD/L)
     4) Oilpricez as last resort
    """
    if requests is None:
        return None
    headers = {"User-Agent": "Mozilla/5.0 (compatible; EVTool/1.1)"}

    # 1) NSW FuelCheck (needs API key)
    api_key = os.getenv("FUELCHECK_API_KEY")
    if state_code in {"NSW", "TAS"} and api_key:
        try:
            # v2 combines NSW+TAS; here we hit NSW and average by product
            url = "https://api.nsw.gov.au/fuelprice/v2/fuel/prices"
            params = {"fuelType": NSW_FUELCHECK_PRODUCT.get(product, 1)}
            resp = requests.get(url, headers={"apikey": api_key, **headers}, params=params, timeout=10)
            if resp.ok:
                data = resp.json()
                prices = [float(p.get("price")) for p in data.get("prices", []) if p.get("price")]
                if prices:
                    return sum(prices) / len(prices) / 100.0  # cents/L -> $/L
        except Exception:
            pass

    # 2) WA FuelWatch
    try:
        # Product=1 is ULP 91. Returns XML/JSON depending on accept header; regex for price
        url = "https://www.fuelwatch.wa.gov.au/fuelwatch/fuelWatchRSS?Product=1"
        resp = requests.get(url, timeout=8, headers=headers)
        if resp.ok:
            # Find all <price>1.789</price>
            matches = re.findall(r"<price>(\d+\.\d+)</price>", resp.text)
            vals = [float(m) for m in matches]
            if vals:
                return sum(vals) / len(vals)
    except Exception:
        pass

    # 3) GlobalPetrolPrices (AUD/L)
    try:
        url = "https://www.globalpetrolprices.com/Australia/gasoline_prices/"
        resp = requests.get(url, timeout=8, headers=headers)
        if resp.ok:
            # Look for 'AUD per liter' or a table row with AUD
            m = re.search(r"Current price.*?([\d\.]+)\s*(?:AUD|A\$|\$)\s*/?\s*(?:per)?\s*liter", resp.text, flags=re.I|re.S)
            if not m:
                # backup: find a row with AUD per liter near 'Australia Gasoline prices'
                m = re.search(r"Australia Gasoline prices.*?AUD[^0-9]*([\d\.]+)\s*per\s*liter", resp.text, flags=re.I|re.S)
            if m:
                return float(m.group(1))
    except Exception:
        pass

    # 4) Oilpricez
    try:
        url = "https://oilpricez.com/au/australia-gasoline-price"
        resp = requests.get(url, timeout=8, headers=headers)
        if resp.ok:
            m = re.search(r"Gasoline.*?Price\s*per\s*Litre.*?([\d\.]+)", resp.text, flags=re.I|re.S)
            if m:
                return float(m.group(1))
    except Exception:
        pass

    return None


# ----------------------------------------
# ICE efficiency (L/100km) fetchers
# ----------------------------------------
def from_local_ice_csv(make: str, model: str, variant: Optional[str] = None, year: Optional[int] = None) -> Optional[float]:
    if not os.path.exists(ICE_LOCAL_CSV):
        return None
    try:
        df = pd.read_csv(ICE_LOCAL_CSV)
        for col in ["make", "model", "variant", "year", "l_per_100km"]:
            if col not in df.columns:
                return None
        q = (df["make"].astype(str).str.lower() == make.lower()) & (df["model"].astype(str).str.lower() == model.lower())
        if variant:
            q &= df["variant"].astype(str).str.lower() == variant.lower()
        if year:
            q &= df["year"].astype(int) == int(year)
        cand = df[q]
        if not cand.empty:
            return float(cand.iloc[0]["l_per_100km"])
    except Exception:
        return None
    return None


def fetch_ice_efficiency(make: str, model: str, variant: Optional[str] = None, year: Optional[int] = None) -> Optional[float]:
    """
    Attempts to return combined-cycle L/100km for an ICE vehicle.
    Tries:
      1) Local CSV (ice_vehicles.csv)
      2) US fueleconomy.gov JSON API (convert MPG -> L/100km) as a pragmatic fallback
         (Note: US cycle; may differ from AU ADR81/02)
    """
    # 1) Local CSV
    eff = from_local_ice_csv(make, model, variant, year)
    if eff:
        return eff

    if requests is None:
        return None

    # 2) US EPA API fallback (approximate)
    try:
        # get list of years for make/model; if none provided, choose latest
        if year is None:
            yresp = requests.get(f"https://www.fueleconomy.gov/ws/rest/vehicle/menu/year", timeout=8)
            years = re.findall(r"<text>(\d{4})</text>", yresp.text) if yresp.ok else []
            years = sorted([int(y) for y in years], reverse=True)
        else:
            years = [int(year)]
        for y in years[:5]:  # try a few recent years
            mresp = requests.get(f"https://www.fueleconomy.gov/ws/rest/vehicle/menu/make?year={y}", timeout=8)
            if not mresp.ok: 
                continue
            makes = re.findall(r"<text>([^<]+)</text>", mresp.text)
            # find closest make
            if not any(m.lower() == make.lower() for m in makes):
                continue
            modresp = requests.get(f"https://www.fueleconomy.gov/ws/rest/vehicle/menu/model?year={y}&make={make}", timeout=8)
            if not modresp.ok:
                continue
            models = re.findall(r"<text>([^<]+)</text>", modresp.text)
            # choose best model match by substring
            target_model = None
            for mm in models:
                if mm.lower() == model.lower() or model.lower() in mm.lower():
                    target_model = mm
                    break
            if not target_model:
                continue
            # get vehicle IDs
            vresp = requests.get(f"https://www.fueleconomy.gov/ws/rest/vehicle/menu/options?year={y}&make={make}&model={target_model}", timeout=8)
            if not vresp.ok:
                continue
            ids = re.findall(r"<value>(\d+)</value>", vresp.text)
            for vid in ids[:3]:
                dresp = requests.get(f"https://www.fueleconomy.gov/ws/rest/vehicle/{vid}", timeout=8)
                if not dresp.ok:
                    continue
                # combined mpg (comb08) or adjusted
                m = re.search(r"<comb08>(\d+)</comb08>", dresp.text)
                if not m:
                    m = re.search(r"<combA08>(\d+)</combA08>", dresp.text)
                if m:
                    mpg = float(m.group(1))
                    if mpg > 0:
                        return 235.215 / mpg  # convert MPG to L/100km
    except Exception:
        pass

    return None


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

    # Select the first matching EV row
    chosen_ev = subset_variant.iloc[0] if not subset_variant.empty else subset_model[subset_model[model_col].astype(str).str.lower() == chosen_model.lower()].iloc[0]

    ev_name = ev_identity(chosen_ev)
    ev_eff = infer_ev_efficiency_kwh_per_km(chosen_ev) or 0.15
    print(f"\nSelected EV: {ev_name}")
    print(f"Inferred efficiency: {ev_eff:.3f} kWh/km")

    # Trip distance
    distance_km = prompt_float("\nEnter trip distance (km)", min_val=0.1)

    # Electricity price source
    print("\nElectricity price source:")
    src_choice = prompt_choice("1=Auto (internet)  2=Manual  3=Average of stations file\nEnter number: ", ["Auto (internet)", "Manual", "Average of stations file"])

    elec_price = 0.30
    price_note = "default"
    if src_choice == 1:
        print("\nChoose your State/Territory:")
        disp = [f"{k} - {STATE_NAMES[k]}" for k in AU_STATES]
        sidx = prompt_choice("Enter number: ", disp)
        state = AU_STATES[sidx - 1]

        prices = fetch_au_elec_prices()
        if state in prices:
            elec_price = prices[state]
            price_note = f"Auto scrape for {state}"
        elif prices:
            elec_price = sum(prices.values()) / len(prices)
            price_note = f"Auto (mean of {len(prices)} states)"
        else:
            print("[WARN] Could not fetch electricity prices. Falling back to $0.30/kWh.")
            elec_price = 0.30
            price_note = "fallback default"
    elif src_choice == 2:
        elec_price = prompt_float("Enter electricity price ($/kWh), e.g., 0.33", min_val=0.05)
        price_note = "manual"
    else:
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

    # ICE efficiency and petrol price
    print("\nICE efficiency source:")
    ice_choice = prompt_choice("1=Fetch online  2=Manual entry\nEnter number: ", ["Fetch online", "Manual entry"])

    ice_l_per_100 = None
    if ice_choice == 1:
        year = None
        # optional: ask for year to improve match
        try:
            y_in = input("Enter vehicle year (or press Enter to skip): ").strip()
            if y_in:
                year = int(y_in)
        except Exception:
            year = None
        ice_l_per_100 = fetch_ice_efficiency(chosen_make, chosen_model, chosen_variant, year)
        if ice_l_per_100 is None:
            print("[WARN] Could not fetch ICE efficiency online. Switching to manual entry.")
    if ice_l_per_100 is None:
        ice_l_per_100 = prompt_float("Enter ICE fuel efficiency (L/100km)", default=7.0, min_val=1.0)

    # Petrol price
    print("\nPetrol price source:")
    p_choice = prompt_choice("1=Auto (internet)  2=Manual\nEnter number: ", ["Auto (internet)", "Manual"])
    if p_choice == 1:
        print("\nChoose your State/Territory for petrol price:")
        disp = [f"{k} - {STATE_NAMES[k]}" for k in AU_STATES]
        sidx = prompt_choice("Enter number: ", disp)
        state = AU_STATES[sidx - 1]
        fuel_price = fetch_petrol_price_au(state_code=state) or 2.0
        if fuel_price == 2.0:
            print("[WARN] Could not fetch petrol price; using $2.00/L default.")
    else:
        fuel_price = prompt_float("Enter petrol price ($/L)", default=2.0, min_val=0.5)

    # Compute
    ev_cost = distance_km * ev_eff * elec_price
    ice_cost = (ice_l_per_100 / 100.0) * distance_km * fuel_price
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
    print(f"Petrol price:           ${fuel_price:.2f}/L")
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
# CLI wrapper
# ----------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EV vs ICE Cost Tool (interactive).")
    parser.add_argument("--interactive", action="store_true", help="Run interactive prompts.")
    args, _ = parser.parse_known_args()

    if args.interactive:
        interactive_flow()
    else:
        print("Run with --interactive for the guided experience.")
        print("Example: python app.py --interactive")


if __name__ == "__main__":
    main()
