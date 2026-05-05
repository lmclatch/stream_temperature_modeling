"""
Audit all 90 basins for data quality issues that affect LSTM training.

Checks per basin:
  - % NaN in discharge (train and validation periods)
  - % NaN in stream_temperature (train and validation periods)
  - % NaN in each ERA5 forcing variable
  - Time index health (unique, freq='h')
  - Range of stream_temperature (catch unphysical values)

Flags basins that should be dropped for:
  - WITH-Q model (any basin with > 50% NaN in discharge during train OR val)
  - NO-Q model (any basin with > 90% NaN in stream_temperature during val,
    or insufficient training T data)

Usage:
    python audit_basins.py [--data-dir DIR] [--out-csv PATH]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    type=Path,
    default=Path(
        "/uufs/chpc.utah.edu/common/home/johnsonrc-group1/hourly_streamtemp/nh_data/time_series"
    ),
)
parser.add_argument(
    "--out-csv",
    type=Path,
    default=Path("/uufs/chpc.utah.edu/common/home/u1566670/hydroinformatics/stream_temperature_modeling/conus/data/basin_audit.csv"),
)
parser.add_argument("--train-start", default="2013-01-01")
parser.add_argument("--train-end", default="2022-12-31")
parser.add_argument("--val-start", default="2023-01-01")
parser.add_argument("--val-end", default="2023-12-31")
# Test period from your config
parser.add_argument("--test-start", default="2024-01-01")
parser.add_argument("--test-end", default="2024-09-30")
args = parser.parse_args()

files = sorted(args.data_dir.glob("*.nc"))
print(f"Auditing {len(files)} basins from {args.data_dir}")
print()

records = []
for f in files:
    bid = f.stem
    ds = xr.open_dataset(f)

    rec = {"basin": bid}

    # Time index health
    idx = pd.DatetimeIndex(ds.date.values)
    rec["n_timesteps"] = len(idx)
    rec["idx_unique"] = idx.is_unique
    rec["idx_freq"] = pd.infer_freq(idx) or "irregular"
    rec["start"] = str(idx[0])
    rec["end"] = str(idx[-1])

    # Per-period NaN fractions for discharge and target
    for period_name, start, end in [
        ("train", args.train_start, args.train_end),
        ("val", args.val_start, args.val_end),
        ("test", args.test_start, args.test_end),
    ]:
        sub = ds.sel(date=slice(start, end))
        if sub.date.size == 0:
            rec[f"{period_name}_n"] = 0
            rec[f"{period_name}_q_nan_pct"] = np.nan
            rec[f"{period_name}_T_nan_pct"] = np.nan
            continue
        rec[f"{period_name}_n"] = int(sub.date.size)

        if "discharge" in sub.data_vars:
            q = sub["discharge"].values
            rec[f"{period_name}_q_nan_pct"] = round(100 * np.isnan(q).sum() / q.size, 1)
        else:
            rec[f"{period_name}_q_nan_pct"] = np.nan

        if "stream_temperature" in sub.data_vars:
            t = sub["stream_temperature"].values
            rec[f"{period_name}_T_nan_pct"] = round(100 * np.isnan(t).sum() / t.size, 1)
        else:
            rec[f"{period_name}_T_nan_pct"] = np.nan

    # T value range (over full record), to catch unphysical outliers
    if "stream_temperature" in ds.data_vars:
        t_all = ds["stream_temperature"].values
        t_clean = t_all[~np.isnan(t_all)]
        if len(t_clean) > 0:
            rec["T_min"] = round(float(t_clean.min()), 2)
            rec["T_max"] = round(float(t_clean.max()), 2)
        else:
            rec["T_min"] = np.nan
            rec["T_max"] = np.nan

    # NaN fraction in any ERA5 forcing (full record) — should be 0 always
    era5_vars = [v for v in ds.data_vars if v not in ("discharge", "stream_temperature")]
    max_era5_nan = 0
    worst_era5 = ""
    for v in era5_vars:
        nanfrac = 100 * np.isnan(ds[v].values).sum() / ds[v].size
        if nanfrac > max_era5_nan:
            max_era5_nan = nanfrac
            worst_era5 = v
    rec["max_era5_nan_pct"] = round(max_era5_nan, 1)
    rec["worst_era5_var"] = worst_era5 if max_era5_nan > 0 else ""

    ds.close()
    records.append(rec)

df = pd.DataFrame(records)

# --- Flag problem basins for each modeling strategy ---
# WITH-Q: drop if Q is mostly missing in train OR val
df["drop_for_withQ"] = (df["train_q_nan_pct"] > 50) | (df["val_q_nan_pct"] > 50)
# NO-Q: drop if T is mostly missing in val (can't evaluate)
df["drop_for_noQ"] = df["val_T_nan_pct"] > 90
# Either model: drop if T is mostly missing in train (can't learn)
df["drop_for_training"] = df["train_T_nan_pct"] > 90
# ERA5 issues: should never be flagged but check anyway
df["drop_for_era5"] = df["max_era5_nan_pct"] > 1
# Index issues: caught earlier but re-check
df["drop_for_index"] = (~df["idx_unique"]) | (df["idx_freq"] != "h")

df["drop_any"] = (
    df["drop_for_withQ"]
    | df["drop_for_noQ"]
    | df["drop_for_training"]
    | df["drop_for_era5"]
    | df["drop_for_index"]
)

# --- Print report ---
print(f"{'='*60}")
print("AUDIT SUMMARY")
print(f"{'='*60}")
print(f"Total basins:        {len(df)}")
print(f"  Index issues:      {df['drop_for_index'].sum()}")
print(f"  ERA5 NaN issues:   {df['drop_for_era5'].sum()}")
print(f"  Train T missing:   {df['drop_for_training'].sum()}")
print(f"  Val T missing:     {df['drop_for_noQ'].sum()}")
print(f"  Q missing (with-Q):{df['drop_for_withQ'].sum()}")
print(f"  ANY issue:         {df['drop_any'].sum()}")
print()
print(f"Basins usable for WITH-Q model: {(~(df['drop_for_withQ'] | df['drop_for_training'] | df['drop_for_noQ'] | df['drop_for_era5'] | df['drop_for_index'])).sum()}")
print(f"Basins usable for NO-Q model:   {(~(df['drop_for_training'] | df['drop_for_noQ'] | df['drop_for_era5'] | df['drop_for_index'])).sum()}")
print()

problem = df[df["drop_any"]].copy()
if len(problem) > 0:
    print("Problem basins:")
    cols_to_show = [
        "basin",
        "train_q_nan_pct",
        "val_q_nan_pct",
        "train_T_nan_pct",
        "val_T_nan_pct",
        "T_min",
        "T_max",
        "drop_for_withQ",
        "drop_for_noQ",
    ]
    print(problem[cols_to_show].to_string(index=False))
print()

# --- Write outputs ---
args.out_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.out_csv, index=False)
print(f"Full audit written to: {args.out_csv}")

# Two basin lists for the two model variants
basin_list_dir = args.out_csv.parent
withq_basins = df.loc[
    ~(
        df["drop_for_withQ"]
        | df["drop_for_training"]
        | df["drop_for_noQ"]
        | df["drop_for_era5"]
        | df["drop_for_index"]
    ),
    "basin",
].tolist()
noq_basins = df.loc[
    ~(
        df["drop_for_training"]
        | df["drop_for_noQ"]
        | df["drop_for_era5"]
        | df["drop_for_index"]
    ),
    "basin",
].tolist()

(basin_list_dir / "basins_withQ.txt").write_text("\n".join(withq_basins) + "\n")
(basin_list_dir / "basins_noQ.txt").write_text("\n".join(noq_basins) + "\n")
print(f"With-Q basin list ({len(withq_basins)}): {basin_list_dir / 'basins_withQ.txt'}")
print(f"No-Q basin list  ({len(noq_basins)}): {basin_list_dir / 'basins_noQ.txt'}")