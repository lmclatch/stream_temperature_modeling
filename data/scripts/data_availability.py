import pandas as pd
import dataretrieval.nwis as nwis
import time
import os

CAMELS_DIR = "/uufs/chpc.utah.edu/common/home/u1566670/civil-group1/CAMELS"
OUTPUT_FILE = "/uufs/chpc.utah.edu/common/home/u1566670/hydroinformatics/stream_temperature_modeling/alaska/camels_temperature_availability_hourly.csv"

MIN_YEARS = 4.0
MAX_PCT_MISSING = 10.0

# --- Load CAMELS gauge IDs ---
camels_names = pd.read_csv(
    f"{CAMELS_DIR}/camels_name.txt",
    sep=";"
)
camels_names["gauge_id"] = camels_names["gauge_id"].astype(str).str.zfill(8)
all_sites = camels_names["gauge_id"].tolist()
print(f"Total CAMELS sites: {len(all_sites)}")

# --- Check temperature availability ---
if os.path.exists(OUTPUT_FILE):
    print("Loading saved results...")
    results_df = pd.read_csv(OUTPUT_FILE, dtype={"site": str})
else:
    results = []

    for i, site in enumerate(all_sites):
        try:
            result = nwis.get_record(
                sites=site,
                service="iv",
                parameterCd="00010",
                start="1980-01-01",
                end="2023-12-31"
            )

            if isinstance(result, tuple):
                site_info = result[0]
            else:
                site_info = result

            if len(site_info) == 0:
                has_temp = False
                n_years = 0.0
                pct_missing = None
            else:
                has_temp = True

                # --- FIX: count distinct calendar days with observations ---
                site_info.index = pd.to_datetime(site_info.index, errors='coerce')
                site_info = site_info[site_info.index.notna()]

                if len(site_info) == 0:
                    has_temp = False
                    n_years = 0.0
                    pct_missing = None
                else:
                    n_distinct_days = site_info.index.normalize().nunique()
                    n_years = round(n_distinct_days / 365.25, 2)

                    temp_col = [c for c in site_info.columns if "00010" in c]
                    if temp_col:
                        pct_missing = round(site_info[temp_col[0]].isna().mean() * 100, 1)
                    else:
                        pct_missing = None

        except Exception as e:
            print(f"  Error on {site}: {e}")
            has_temp = False
            n_years = 0.0
            pct_missing = None

        results.append({
            "site": site,
            "has_temp": has_temp,
            "n_years": n_years,
            "pct_missing": pct_missing
        })

        if i % 10 == 0:
            print(f"  {i}/{len(all_sites)} — {site} — has_temp: {has_temp}, n_years: {n_years}")
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

        time.sleep(0.5)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done. Saved to {OUTPUT_FILE}")

# --- Summary ---
has_temp_df = results_df[results_df["has_temp"] == True].copy()
print(f"\nTotal CAMELS sites checked: {len(results_df)}")
print(f"Sites with any temperature data: {len(has_temp_df)}")
print()
print("Distribution of available record length (years):")
print(has_temp_df["n_years"].describe())
print()
print("Basin count at different thresholds:")
for thresh in [2, 3, 4, 5, 7, 10]:
    n = (has_temp_df["n_years"] >= thresh).sum()
    print(f"  n_years >= {thresh}: {n} basins")

# --- Apply filter ---
passing = has_temp_df[
    (has_temp_df["n_years"] >= MIN_YEARS) &
    (has_temp_df["pct_missing"] < MAX_PCT_MISSING)
]
print(f"\nSites passing filter ({MIN_YEARS}+ years, <{MAX_PCT_MISSING}% missing): {len(passing)}")