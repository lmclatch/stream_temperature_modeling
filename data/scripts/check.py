"""
Audit basins under multiple candidate train/val/test split windows.

For each candidate split, reports:
  - How many basins have clean Q in train and val
  - How many basins have clean T in train, val, and test
  - How many basins survive both filters (= usable for with-Q model)
  - Same for no-Q model (only T constraint matters)

The goal is to pick a split that keeps the most basins viable.

Usage:
    python audit_splits.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

DATA_DIR = Path(
    "/uufs/chpc.utah.edu/common/home/johnsonrc-group1/hourly_streamtemp/nh_data/time_series"
)

# Candidate splits to evaluate. Add or remove as you like.
CANDIDATES = [
    {
        "name": "current (2015-22 / 2023 / 2024Q1-Q3)",
        "train": ("2015-01-01", "2022-12-31"),
        "val": ("2023-01-01", "2023-12-31"),
        "test": ("2024-01-01", "2024-09-30"),
    },
    {
        "name": "longer val (2012-22 / 2023 / 2024Q1-Q3)",
        "train": ("2012-01-01", "2022-12-31"),
        "val": ("2023-01-01", "2023-12-31"),
        "test": ("2024-01-01", "2024-09-30"),
    },
    {
        "name": "10yr train, 2yr val, 3yr test (2010-19 / 2020-21 / 2022-24)",
        "train": ("2010-01-01", "2019-12-31"),
        "val": ("2020-01-01", "2021-12-31"),
        "test": ("2022-01-01", "2024-12-31"),
    },
    {
        "name": "20yr train, 2yr val, 4yr test (2000-19 / 2020-21 / 2022-25)",
        "train": ("2000-01-01", "2019-12-31"),
        "val": ("2020-01-01", "2021-12-31"),
        "test": ("2022-01-01", "2025-12-31"),
    },
    {
        "name": "long train through covid, 2yr val (2000-21 / 2022-23 / 2024-25)",
        "train": ("2000-01-01", "2021-12-31"),
        "val": ("2022-01-01", "2023-12-31"),
        "test": ("2024-01-01", "2025-12-31"),
    },
]

# Thresholds to call a basin "clean" in a given period
Q_NAN_THRESH = 50  # basin fails if > this % Q is NaN
T_NAN_THRESH = 90  # basin fails if > this % T is NaN
T_TRAIN_THRESH = 70  # need at least 30% T data in train to learn

files = sorted(DATA_DIR.glob("*.nc"))
print(f"Auditing {len(files)} basins across {len(CANDIDATES)} candidate splits\n")

# Pre-load all basin data once (faster than reopening per candidate)
basin_data = {}
for f in files:
    ds = xr.open_dataset(f)
    basin_data[f.stem] = {
        "Q": ds["discharge"].to_pandas() if "discharge" in ds.data_vars else None,
        "T": ds["stream_temperature"].to_pandas()
        if "stream_temperature" in ds.data_vars
        else None,
    }
    ds.close()
print(f"Loaded {len(basin_data)} basin records into memory\n")


def nan_pct(series, start, end):
    sub = series.loc[start:end]
    if len(sub) == 0:
        return 100.0
    return 100 * sub.isna().sum() / len(sub)


# Per-candidate per-basin breakdown
all_results = []
for cand in CANDIDATES:
    results = []
    for bid, dat in basin_data.items():
        q = dat["Q"]
        t = dat["T"]
        rec = {"basin": bid, "split": cand["name"]}

        if q is not None:
            rec["train_q_nan"] = round(nan_pct(q, *cand["train"]), 1)
            rec["val_q_nan"] = round(nan_pct(q, *cand["val"]), 1)
            rec["test_q_nan"] = round(nan_pct(q, *cand["test"]), 1)
        else:
            rec["train_q_nan"] = rec["val_q_nan"] = rec["test_q_nan"] = 100.0

        if t is not None:
            rec["train_t_nan"] = round(nan_pct(t, *cand["train"]), 1)
            rec["val_t_nan"] = round(nan_pct(t, *cand["val"]), 1)
            rec["test_t_nan"] = round(nan_pct(t, *cand["test"]), 1)
        else:
            rec["train_t_nan"] = rec["val_t_nan"] = rec["test_t_nan"] = 100.0

        # Eligibility flags
        # T must be present enough to train AND to evaluate
        rec["t_ok"] = (
            rec["train_t_nan"] < T_TRAIN_THRESH
            and rec["val_t_nan"] < T_NAN_THRESH
            and rec["test_t_nan"] < T_NAN_THRESH
        )
        # Q must be present enough in train and val to learn from / evaluate
        rec["q_ok"] = (
            rec["train_q_nan"] < Q_NAN_THRESH and rec["val_q_nan"] < Q_NAN_THRESH
        )
        rec["usable_withQ"] = rec["t_ok"] and rec["q_ok"]
        rec["usable_noQ"] = rec["t_ok"]

        results.append(rec)

    df_cand = pd.DataFrame(results)
    all_results.append(df_cand)

    # Summary
    print(f"=== {cand['name']} ===")
    print(f"  Train: {cand['train'][0]} to {cand['train'][1]}")
    print(f"  Val:   {cand['val'][0]} to {cand['val'][1]}")
    print(f"  Test:  {cand['test'][0]} to {cand['test'][1]}")
    print(f"  Basins clean for T (train+val+test):    {df_cand['t_ok'].sum():>3} / {len(df_cand)}")
    print(f"  Basins clean for Q (train+val):         {df_cand['q_ok'].sum():>3} / {len(df_cand)}")
    print(f"  Usable for WITH-Q model:                {df_cand['usable_withQ'].sum():>3} / {len(df_cand)}")
    print(f"  Usable for NO-Q model:                  {df_cand['usable_noQ'].sum():>3} / {len(df_cand)}")
    print()

# # Concatenate full table to disk
# out_dir = Path(
#     "/uufs/chpc.utah.edu/common/home/u1566670/hydroinformatics/stream_temperature_modeling/conus/results"
# )
# out_dir.mkdir(parents=True, exist_ok=True)
# big = pd.concat(all_results, ignore_index=True)
# big.to_csv(out_dir / "split_audit.csv", index=False)
# print(f"Full per-basin/per-split table: {out_dir / 'split_audit.csv'}")

# # For each candidate, also dump the basin lists
# for df_cand, cand in zip(all_results, CANDIDATES):
#     safe_name = cand["name"].split(" ")[0]  # 'current', 'longer', '10yr', etc.
#     withq = df_cand[df_cand["usable_withQ"]]["basin"].tolist()
#     noq = df_cand[df_cand["usable_noQ"]]["basin"].tolist()
#     (out_dir / f"basins_withQ_{safe_name}.txt").write_text("\n".join(withq) + "\n")
#     (out_dir / f"basins_noQ_{safe_name}.txt").write_text("\n".join(noq) + "\n")

# print(f"Basin lists written to: {out_dir}/basins_*.txt")