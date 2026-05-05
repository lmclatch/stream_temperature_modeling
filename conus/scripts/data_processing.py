"""
Build a NeuralHydrology GenericDataset-compatible directory from the raw
hourly stream-temperature data.

Inputs (hardcoded, see paths below):
  hourly/era5_forcing/time_series_deaccumulated/<basin>.nc   — 14 ERA5-Land vars
  hourly/streamflow/<basin>.csv                              — discharge_cfs
  hourly/stream_temperature/<basin>.csv                      — temperature_C
  hourly/attributes/camels_attributes_90basins.csv           — 59 CAMELS attrs

Output layout (what NH GenericDataset reads):
  nh_data/
    time_series/<basin>.nc       — 14 ERA5 vars + discharge + stream_temperature,
                                   on the ERA5 hourly time axis, coord called `date`
    attributes/attributes.csv    — numeric CAMELS attrs, indexed by `basin`
    basins_all.txt               — list of basins that built successfully
    qc_stats.csv                 — per-basin QC: hours, T obs total + by year


Usage:
    python 01_preprocess_to_generic.py \\
        --out-dir /uufs/chpc.utah.edu/common/home/johnsonrc-group1/hourly_streamtemp/nh_data
    # add --limit 5 to test on 5 basins first
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path("/uufs/chpc.utah.edu/common/home/johnsonrc-group1/hourly_streamtemp/hourly")
FORCING_DIR = ROOT / "era5_forcing" / "time_series_deaccumulated"
Q_DIR = ROOT / "streamflow"
T_DIR = ROOT / "stream_temperature"
ATTR_FILE = ROOT / "attributes" / "camels_attributes_90basins.csv"


def find_time_coord(ds: xr.Dataset) -> str:
    """ERA5 files may use 'time', 'valid_time', or 'date' — find whichever exists."""
    for cand in ("date", "time", "valid_time", "Time"):
        if cand in ds.coords or cand in ds.dims:
            return cand
    raise KeyError(f"No recognizable time coord in {list(ds.coords)}")


def load_forcing(basin: str) -> xr.Dataset:
    """Load ERA5 NetCDF; rename time coord to `date`; squeeze any singleton dims."""
    ds = xr.open_dataset(FORCING_DIR / f"{basin}.nc")
    tcoord = find_time_coord(ds)
    if tcoord != "date":
        ds = ds.rename({tcoord: "date"})
    for d in list(ds.dims):
        if d != "date" and ds.sizes[d] == 1:
            ds = ds.squeeze(d, drop=True)
    return ds


def load_csv_series(path: Path, value_col: str, new_name: str) -> xr.DataArray:
    """Load a 2-col CSV (datetime, value) into an xarray DataArray indexed by `date`."""
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()
    # Drop duplicate timestamps if any (keep first occurrence)
    df = df[~df.index.duplicated(keep="first")]
    s = df[value_col].astype("float32")
    s.index.name = "date"
    return s.to_xarray().rename(new_name)


def build_basin_nc(basin: str, out_dir: Path) -> dict:
    """Merge forcing + Q + T into one NetCDF on ERA5's hourly time axis.
    Returns QC stats dict for this basin.
    """
    forcing = load_forcing(basin)
    q = load_csv_series(Q_DIR / f"{basin}.csv", "discharge_cfs", "discharge")
    t = load_csv_series(T_DIR / f"{basin}.csv", "temperature_C", "stream_temperature")

    # Reindex Q and T onto forcing time axis. Hours outside ERA5 coverage are dropped;
    # hours where forcing exists but T/Q don't become NaN (NH skips these targets).
    q = q.reindex(date=forcing["date"])
    t = t.reindex(date=forcing["date"])

    merged = forcing.assign(discharge=q, stream_temperature=t)
    merged["date"] = merged["date"].astype("datetime64[ns]")

    # Force a clean, regular hourly axis. Some ERA5 source files have a duplicate
    # timestamp + 1-hour gap (DST artifact); NH rejects such indices because
    # pandas.infer_freq() returns None on irregular series.
    # First drop any duplicates (keep first), then reindex onto a regular range.
    _, unique_idx = np.unique(merged["date"].values, return_index=True)
    if len(unique_idx) < merged.sizes["date"]:
        merged = merged.isel(date=np.sort(unique_idx))
    clean_axis = pd.date_range(
        start=pd.Timestamp(merged["date"].values[0]),
        end=pd.Timestamp(merged["date"].values[-1]),
        freq="1h",
    )
    merged = merged.reindex(date=clean_axis)

    out_path = out_dir / "time_series" / f"{basin}.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4} for v in merged.data_vars}
    merged.to_netcdf(out_path, encoding=encoding)

    # QC: T observation counts per year, useful for picking train/val/test windows later
    t_series = t.to_series().dropna()
    yearly_counts = t_series.groupby(t_series.index.year).size().to_dict()

    return {
        "basin": basin,
        "n_hours": int(merged.sizes["date"]),
        "ts_start": str(pd.Timestamp(merged["date"].values[0])),
        "ts_end": str(pd.Timestamp(merged["date"].values[-1])),
        "n_t_obs": int(t.notnull().sum().item()),
        "n_q_obs": int(q.notnull().sum().item()),
        "t_first": str(t_series.index[0]) if len(t_series) else "",
        "t_last": str(t_series.index[-1]) if len(t_series) else "",
        # Yearly T counts spread across columns (limit to 2010-2025 for readability)
        **{f"t_obs_{y}": yearly_counts.get(y, 0) for y in range(2010, 2026)},
    }


def build_attributes(out_dir: Path, basins: list[str]):
    """Reformat CAMELS attributes CSV: 8-digit basin IDs, numeric only, indexed by `basin`."""
    attrs = pd.read_csv(ATTR_FILE, dtype={"gauge_id": str})
    # CSV stored gauge_id as int, leading zeros stripped — pad back to 8 digits
    attrs["gauge_id"] = attrs["gauge_id"].str.zfill(8)
    attrs = attrs[attrs["gauge_id"].isin(basins)].copy()
    attrs = attrs.rename(columns={"gauge_id": "basin"}).set_index("basin")

    # Drop non-numeric columns (gauge_name, dom_land_cover, geol_*_class, huc_02,
    # *_prec_timing). NH static features only support numeric values.
    non_numeric = attrs.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  Dropping non-numeric attribute columns ({len(non_numeric)}): {non_numeric}")
        attrs = attrs.drop(columns=non_numeric)

    out = out_dir / "attributes"
    out.mkdir(parents=True, exist_ok=True)
    attrs.to_csv(out / "attributes.csv")
    print(f"  Wrote attributes/attributes.csv: {len(attrs)} basins x {attrs.shape[1]} numeric attrs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output GenericDataset directory")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process first N basins (for testing the pipeline)")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find basins with all 3 sources
    forcing_ids = {f.stem for f in FORCING_DIR.glob("*.nc")}
    q_ids = {f.stem for f in Q_DIR.glob("*.csv")}
    t_ids = {f.stem for f in T_DIR.glob("*.csv")}
    common = sorted(forcing_ids & q_ids & t_ids)
    if args.limit:
        common = common[: args.limit]
    print(f"Processing {len(common)} basins (with forcing + Q + T)\n")

    qc_records = []
    failures = []
    for i, basin in enumerate(common, 1):
        try:
            rec = build_basin_nc(basin, out_dir)
            qc_records.append(rec)
            if i % 10 == 0 or i == len(common):
                print(f"  [{i}/{len(common)}] {basin}: "
                      f"{rec['n_hours']:,}h, T obs={rec['n_t_obs']:,} "
                      f"({rec['t_first'][:10]} → {rec['t_last'][:10]})")
        except Exception as e:
            print(f"  [FAIL] {basin}: {type(e).__name__}: {e}", file=sys.stderr)
            failures.append((basin, str(e)))

    qc_df = pd.DataFrame(qc_records)
    qc_df.to_csv(out_dir / "qc_stats.csv", index=False)
    print(f"\nQC stats written to {out_dir / 'qc_stats.csv'}")

    successful_basins = qc_df["basin"].tolist()
    build_attributes(out_dir, successful_basins)

    (out_dir / "basins_all.txt").write_text("\n".join(successful_basins) + "\n")
    print(f"Wrote basins_all.txt: {len(successful_basins)} basins")

    # Top-line summary
    print("\n=== Summary ===")
    print(f"Basins built:       {len(successful_basins)}/{len(common)}")
    print(f"Failures:           {len(failures)}")
    print(f"Total T obs:        {qc_df['n_t_obs'].sum():,}")
    print(f"Median T obs/basin: {int(qc_df['n_t_obs'].median()):,}")
    train_cols = [f"t_obs_{y}" for y in range(2015, 2023)]
    val_col, test_col = "t_obs_2023", "t_obs_2024"
    qc_df["t_obs_train_window"] = qc_df[train_cols].sum(axis=1)
    print(f"Basins with T obs in train window 2015-2022: "
          f"{(qc_df['t_obs_train_window'] > 0).sum()}")
    print(f"Basins with T obs in val window 2023:        "
          f"{(qc_df[val_col] > 0).sum()}")
    print(f"Basins with T obs in test window 2024:       "
          f"{(qc_df[test_col] > 0).sum()}")

    if failures:
        print(f"\n{len(failures)} basins failed; see stderr.")


if __name__ == "__main__":
    main()