#!/bin/bash
#SBATCH --job-name=era5_90basins
#SBATCH --account=johnsonrc
#SBATCH --partition=kingspeak
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/era5_%j.out
#SBATCH --error=logs/era5_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u1566670@utah.edu

set -eo pipefail

# -------- paths (edit if needed) --------
USER_ID=u1566670
SRC_DIR=/uufs/chpc.utah.edu/common/home/johnsonrc-group1/CAMELS_hourly/era5_land
SHP_ARCHIVE=${SRC_DIR}/CAMELSH_shapefile.7z
BASIN_LIST=/uufs/chpc.utah.edu/common/home/${USER_ID}/hydroinformatics/stream_temperature_modeling/alaska/camels_temperature_availability_hourly.csv
SCRATCH=/scratch/general/vast/${USER_ID}/era5_work
OUTDIR=/uufs/chpc.utah.edu/common/home/johnsonrc-group1/CAMELS_hourly/era5_90basins
# ----------------------------------------

YEAR_START=1980
YEAR_END=2024

mkdir -p "$SCRATCH" "$OUTDIR" logs
ml miniconda3/25.9.1
source activate torch310env

echo "=========================================="
echo "Job:      $SLURM_JOB_ID"
echo "Host:     $(hostname)"
echo "Started:  $(date)"
echo "Scratch:  $SCRATCH"
echo "Output:   $OUTDIR"
echo "Years:    $YEAR_START - $YEAR_END"
echo "=========================================="
df -h "$SCRATCH" | tail -1

export SRC_DIR SHP_ARCHIVE BASIN_LIST SCRATCH OUTDIR YEAR_START YEAR_END

python - <<'PYEOF'
import os
import shutil
import time
from pathlib import Path

import py7zr
import pandas as pd
import xarray as xr
import geopandas as gpd

SRC_DIR     = Path(os.environ["SRC_DIR"])
SHP_ARCHIVE = Path(os.environ["SHP_ARCHIVE"])
BASIN_LIST  = Path(os.environ["BASIN_LIST"])
SCRATCH     = Path(os.environ["SCRATCH"])
OUTDIR      = Path(os.environ["OUTDIR"])
YEAR_START  = int(os.environ["YEAR_START"])
YEAR_END    = int(os.environ["YEAR_END"])

# ---- 1. load basin list ----
df = pd.read_csv(BASIN_LIST, dtype={"site": str})
keep = df[(df["has_temp"] == True) & (df["n_years"] >= 4.0) & (df["pct_missing"] < 10.0)]
my_basins = sorted(keep["site"].str.zfill(8).unique())
print(f"[setup] Basins to extract: {len(my_basins)}", flush=True)

# ---- 2. build gauge_id -> watershed_id (row index) mapping from shapefile ----
shp_dir = SCRATCH / "shapefile"
shp_dir.mkdir(parents=True, exist_ok=True)
print(f"[setup] Extracting shapefile ...", flush=True)
with py7zr.SevenZipFile(SHP_ARCHIVE, "r") as z:
    z.extractall(path=shp_dir)

shp_path = next(shp_dir.rglob("*.shp"))
gdf = gpd.read_file(shp_path)
gdf["GAGE_ID"] = gdf["GAGE_ID"].astype(str).str.zfill(8)
id_to_idx = dict(zip(gdf["GAGE_ID"], gdf.index))

present = [b for b in my_basins if b in id_to_idx]
missing = [b for b in my_basins if b not in id_to_idx]
if missing:
    print(f"[warn] {len(missing)} basins not in shapefile: {missing}", flush=True)
basin_idxs = [id_to_idx[b] for b in present]
print(f"[setup] Mapped {len(present)}/{len(my_basins)} basins to row indices", flush=True)

# ---- 3. output layout: NeuralHydrology GenericDataset ----
#   $OUTDIR/time_series/<gauge_id>.nc
ts_dir = OUTDIR / "time_series"
ts_dir.mkdir(parents=True, exist_ok=True)

# scratch: per-basin yearly pieces, concatenated at the end
pieces_dir = SCRATCH / "pieces"
pieces_dir.mkdir(parents=True, exist_ok=True)

# ---- 4. main loop: extract -> slice -> save pieces -> delete unzipped ----
for year in range(YEAR_START, YEAR_END + 1):
    t0 = time.time()
    archive = SRC_DIR / f"CAMELSH_ERA5_Land_{year}.7z"
    if not archive.exists():
        print(f"[{year}] MISSING archive, skipping: {archive}", flush=True)
        continue

    year_dir = SCRATCH / "unzipped" / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{year}] extracting ...", flush=True)
    with py7zr.SevenZipFile(archive, "r") as z:
        z.extractall(path=year_dir)

    nc_path = next(year_dir.rglob("*.nc"))

    try:
        ds = xr.open_dataset(nc_path)
        sub = ds.isel(watershed_id=basin_idxs)
        # drop the leftover coord along the original unsliced dim if present
        sub = sub.drop_vars([v for v in ["watershed"] if v in sub.coords])
        # attach gauge_id as a coord along watershed_id
        sub = sub.assign_coords(gauge_id=("watershed_id", present))

        # write one small file per basin per year
        for i, b in enumerate(present):
            piece = pieces_dir / b / f"{year}.nc"
            piece.parent.mkdir(parents=True, exist_ok=True)
            one = sub.isel(watershed_id=i).drop_vars(["gauge_id"])
            one.to_netcdf(piece)
        ds.close()
    finally:
        shutil.rmtree(year_dir, ignore_errors=True)  # immediately free ~5 GB

    dt = time.time() - t0
    print(f"[{year}] done ({dt:.0f}s)", flush=True)

# ---- 5. concatenate per-basin yearly pieces -> per-basin final NetCDFs ----
print("\n[concat] combining yearly pieces per basin ...", flush=True)
for b in present:
    bdir = pieces_dir / b
    if not bdir.exists():
        print(f"[concat] {b}: no pieces, skipping"); continue
    pieces = sorted(bdir.glob("*.nc"))
    ds_list = [xr.open_dataset(p) for p in pieces]
    ds = xr.concat(ds_list, dim="time").sortby("time")
    # add gauge_id as an attribute for traceability
    ds.attrs["gauge_id"] = b
    ds.attrs["source"] = "CAMELSH ERA5-Land, subset to 90 basins"
    out = ts_dir / f"{b}.nc"
    ds.to_netcdf(out)
    ds.close()

# ---- 6. clean scratch ----
shutil.rmtree(pieces_dir, ignore_errors=True)
shutil.rmtree(shp_dir, ignore_errors=True)

print("\n[done] all basins written to:", ts_dir, flush=True)
PYEOF

echo "=========================================="
echo "Finished: $(date)"
echo "Output size:"
du -sh "$OUTDIR"
ls "$OUTDIR/time_series" | wc -l
echo "=========================================="

