"""
Inspect data structure before preprocessing. Run this first.
Tells us:
  - ERA5 NetCDF dimension/coordinate names
  - ERA5 variable names (the 14 forcings)
  - Whether streamflow/temperature CSVs parse cleanly
  - Date ranges and overlap per basin
  - List of basins with all 3 data sources present
"""
from pathlib import Path
import pandas as pd
import xarray as xr

ROOT = Path("/uufs/chpc.utah.edu/common/home/johnsonrc-group1/hourly_streamtemp/hourly")
FORCING_DIR = ROOT / "era5_forcing" / "time_series_deaccumulated"
Q_DIR = ROOT / "streamflow"
T_DIR = ROOT / "stream_temperature"
ATTR_FILE = ROOT / "attributes" / "camels_attributes_90basins.csv"


def main():
    # --- ERA5: peek at one file ---
    forcing_files = sorted(FORCING_DIR.glob("*.nc"))
    print(f"ERA5 forcing files: {len(forcing_files)}")
    sample = forcing_files[0]
    print(f"\n=== Inspecting {sample.name} ===")
    ds = xr.open_dataset(sample)
    print(ds)
    print(f"\nDimensions: {dict(ds.dims)}")
    print(f"Coords: {list(ds.coords)}")
    print(f"Data vars: {list(ds.data_vars)}")
    # Try to identify the time coordinate
    time_candidates = [c for c in ds.coords if "time" in c.lower() or "date" in c.lower()]
    print(f"Likely time coord: {time_candidates}")
    ds.close()

    # --- Streamflow ---
    q_files = sorted(Q_DIR.glob("*.csv"))
    print(f"\nStreamflow files: {len(q_files)}")
    q_sample = pd.read_csv(q_files[0], nrows=5)
    print(f"Q columns: {list(q_sample.columns)}")
    print(q_sample)

    # --- Stream temperature ---
    t_files = sorted(T_DIR.glob("*.csv"))
    print(f"\nTemperature files: {len(t_files)}")
    t_sample = pd.read_csv(t_files[0], nrows=5)
    print(f"T columns: {list(t_sample.columns)}")
    print(t_sample)

    # --- Attributes ---
    attrs = pd.read_csv(ATTR_FILE)
    print(f"\nAttributes: {attrs.shape}, gauge_id dtype = {attrs['gauge_id'].dtype}")
    print(f"First 5 gauge_ids: {attrs['gauge_id'].head().tolist()}")

    # --- Cross-reference: which basins have all 3? ---
    forcing_ids = {f.stem for f in forcing_files}
    q_ids = {f.stem for f in q_files}
    t_ids = {f.stem for f in t_files}
    common = forcing_ids & q_ids & t_ids
    print(f"\nBasins with all 3 (forcing, Q, T): {len(common)}")
    missing_q = forcing_ids - q_ids
    missing_t = forcing_ids - t_ids
    if missing_q:
        print(f"  Missing Q ({len(missing_q)}): {sorted(missing_q)[:5]}...")
    if missing_t:
        print(f"  Missing T ({len(missing_t)}): {sorted(missing_t)[:5]}...")

    # --- Date range / record length per basin (sample 5) ---
    print("\n=== Date ranges (first 5 basins with all data) ===")
    for bid in sorted(common)[:5]:
        t_df = pd.read_csv(T_DIR / f"{bid}.csv", parse_dates=["datetime"])
        q_df = pd.read_csv(Q_DIR / f"{bid}.csv", parse_dates=["datetime"])
        t_df = t_df.dropna(subset=["temperature_C"])
        n_years = (t_df["datetime"].max() - t_df["datetime"].min()).days / 365.25
        print(f"  {bid}: T {t_df['datetime'].min()} → {t_df['datetime'].max()} "
              f"({len(t_df):,} obs, ~{n_years:.1f} yr); "
              f"Q {q_df['datetime'].min()} → {q_df['datetime'].max()}")


if __name__ == "__main__":
    main()