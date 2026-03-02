import pandas as pd
import dataretrieval.nwis as nwis
import time
import os

CAMELS_DIR = "/uufs/chpc.utah.edu/common/home/u1566670/civil-group1/CAMELS"
OUTPUT_FILE = "/uufs/chpc.utah.edu/common/home/u1566670/civil-group1/McLatchy/alaska/camels_temperature_availability_hourly.csv"

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
    completed = set()

    for i, site in enumerate(all_sites):
        if site in completed:
            continue
        try:
            result = nwis.get_record(
                sites=site,
                service="iv",           # insantaneous values
                parameterCd="00010",    # stream temperature
                start="1980-01-01",
                end="2023-12-31"
            )

            # Handle variable return types
            if isinstance(result, tuple):
                site_info = result[0]
            else:
                site_info = result

            if len(site_info) == 0:
                has_temp = False
                n_years = 0
                pct_missing = None
            else:
                has_temp = True
                n_years = round(len(site_info) / 365, 1)
                temp_col = [c for c in site_info.columns if "00010" in c]
                if temp_col:
                    pct_missing = round(site_info[temp_col[0]].isna().mean() * 100, 1)
                else:
                    pct_missing = None

        except Exception as e:
            print(f"  Error on {site}: {e}")
            has_temp = False
            n_years = 0
            pct_missing = None

        results.append({
            "site": site,
            "has_temp": has_temp,
            "n_years": n_years,
            "pct_missing": pct_missing
        })

        # Progress + incremental save
        if i % 10 == 0:
            print(f"  {i}/{len(all_sites)} — {site} — has_temp: {has_temp}")
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

        time.sleep(0.5)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done. Saved to {OUTPUT_FILE}")

# --- Summary ---
has_temp = results_df[results_df["has_temp"] == True]
has_temp["n_years_corrected"] = has_temp["n_years"] / (35040 / 365)
print(f"\nSites with temperature data: {len(has_temp)} / {len(results_df)}")
print(f"Sites with 10+ years of data: {len(has_temp[has_temp['n_years'] >= 10])}")
print(f"Sites with 20+ years of data: {len(has_temp[has_temp['n_years'] >= 20])}")

