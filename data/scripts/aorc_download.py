import os
import requests
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from tqdm import tqdm

'''
Downloads and organizes AORC atmospheric forcing data for all 671 CAMELS basins
from HydroShare for use in LSTM-based streamflow prediction.

Source:
    Frame, J. et al. (2025). AORC atmospheric forcing data across CAMELS US basins,
    1980-2024. HydroShare. https://doi.org/10.4211/hs.c738c05278a34bc9848dd14d61cffab9

Output structure (one NetCDF per variable, wide format):
    camels_aorc/
        processed/
            AORC_CAMELS_APCP_surface_1980_2024_hourly.nc
            AORC_CAMELS_TMP_2maboveground_1980_2024_hourly.nc
            ... (8 variables total)
            basin_metadata.csv

Each NetCDF has:
    dims:   time (hourly), basin_id (671 CAMELS gauge IDs)
    coords: time, basin_id, latitude, longitude (basin centroids)

LSTM-ready: after this script, each variable file can be stacked into a
    (time, basin, variable) tensor with xr.open_mfdataset or np.stack.

Known issue in source data:
    Basin-wide aggregation assumed equal-area divides — small effect on values.
    Check HydroShare resource page for corrected version before publication.
'''

# =============================================================================
# USER PARAMETERS
# =============================================================================

# Output directory on CHPC
output_dir = Path('/uufs/chpc.utah.edu/common/home/johnsonrc-group1/camels_aorc')

# HydroShare resource ID
HS_RESOURCE_ID = 'c738c05278a34bc9848dd14d61cffab9'

# All 8 AORC variables
VARIABLES = [
    'APCP_surface',           # Precipitation         (mm)
    'TMP_2maboveground',      # Temperature           (K)
    'UGRD_10maboveground',    # Wind U-component      (m/s)
    'VGRD_10maboveground',    # Wind V-component      (m/s)
    'DSWRF_surface',          # Shortwave radiation   (W/m²)
    'DLWRF_surface',          # Longwave radiation    (W/m²)
    'SPFH_2maboveground',     # Specific humidity     (kg/kg)
    'PRES_surface',           # Surface pressure      (Pa)
]

# =============================================================================
# DIRECTORY SETUP
# =============================================================================

raw_dir       = output_dir / 'raw'        # downloaded files land here
processed_dir = output_dir / 'processed'  # wide-format NetCDFs go here

raw_dir.mkdir(parents=True, exist_ok=True)
processed_dir.mkdir(parents=True, exist_ok=True)

print(f"Output root : {output_dir}")
print(f"Raw data    : {raw_dir}")
print(f"Processed   : {processed_dir}\n")

# =============================================================================
# STEP 1: DISCOVER FILES ON HYDROSHARE
# HydroShare exposes a REST API that lists all files in a resource.
# The AORC CAMELS dataset is organized as one NetCDF per basin (gauge ID).
# =============================================================================

def get_hydroshare_file_list(resource_id):
    """Return list of (filename, download_url) tuples for a HydroShare resource."""
    api_url = f"https://www.hydroshare.org/hsapi/resource/{resource_id}/files/"
    files = []
    url = api_url
    while url:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for f in data.get('results', []):
            fname = f['file_name']
            furl  = f['url']
            files.append((fname, furl))
        url = data.get('next')   # paginate through all files
    return files

print("Querying HydroShare file list ...")
try:
    all_files = get_hydroshare_file_list(HS_RESOURCE_ID)
    nc_files  = [(fname, url) for fname, url in all_files if fname.endswith('.nc')]
    print(f"Found {len(nc_files)} NetCDF files.\n")
except Exception as e:
    print(f"ERROR fetching file list: {e}")
    print("Check that the HydroShare resource is public and the resource ID is correct.")
    print(f"Resource page: https://www.hydroshare.org/resource/{HS_RESOURCE_ID}/")
    raise

# =============================================================================
# STEP 2: DOWNLOAD RAW FILES (one NetCDF per basin)
# Skips files already downloaded — safe to re-run after interruption.
# =============================================================================

def download_file(url, dest_path, chunk_size=1024*1024):
    """Stream-download a file with a progress bar. Skip if already exists."""
    if dest_path.exists():
        return  # already downloaded
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f, tqdm(
        total=total, unit='B', unit_scale=True,
        desc=dest_path.name, leave=False
    ) as bar:
        for chunk in resp.iter_content(chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

print(f"Downloading {len(nc_files)} basin files to {raw_dir} ...")
print("(Files already downloaded will be skipped)\n")

for fname, furl in tqdm(nc_files, desc="Basins downloaded"):
    download_file(furl, raw_dir / fname)

print("\nAll downloads complete.\n")

# =============================================================================
# STEP 3: LOAD ALL BASIN FILES AND BUILD WIDE-FORMAT NetCDFs
#
# Each per-basin NetCDF has dims: (time,) and variables for each forcing.
# We concatenate across basins to produce:
#   dims:   (time, basin_id)
#   for each variable: DataArray of shape (time, basin_id)
#
# Output: one NetCDF per variable, wide format, basins as a dimension.
# This is the most LSTM-friendly layout:
#   - Load all variables → stack → shape (time, 671, 8) tensor
# =============================================================================

print("Loading basin files and building wide-format dataset ...")

basin_files = sorted(raw_dir.glob('*.nc'))
print(f"Found {len(basin_files)} basin NetCDF files in {raw_dir}\n")

# --- Collect per-basin DataArrays ---
basin_ids  = []
lat_list   = []
lon_list   = []
var_arrays = {v: [] for v in VARIABLES}

for bf in tqdm(basin_files, desc="Reading basins"):
    # Extract gauge ID from filename (expected format: <gauge_id>.nc or similar)
    gauge_id = bf.stem   # filename without extension

    ds = xr.open_dataset(bf)

    # Grab basin centroid coords if available (common in CAMELS-style files)
    lat = float(ds.attrs.get('gauge_lat', ds.attrs.get('lat', np.nan)))
    lon = float(ds.attrs.get('gauge_lon', ds.attrs.get('lon', np.nan)))

    basin_ids.append(gauge_id)
    lat_list.append(lat)
    lon_list.append(lon)

    for var in VARIABLES:
        if var in ds:
            var_arrays[var].append(ds[var])
        else:
            # Variable missing for this basin — fill with NaN at same time coords
            time_coord = ds['time'] if 'time' in ds.coords else None
            if time_coord is not None:
                nan_da = xr.full_like(ds[list(ds.data_vars)[0]], fill_value=np.nan)
                nan_da.name = var
                var_arrays[var].append(nan_da)
            print(f"  WARNING: '{var}' missing in {bf.name}")

    ds.close()

print(f"\nLoaded {len(basin_ids)} basins.")
print(f"Variables found: {[v for v in VARIABLES if var_arrays[v]]}\n")

# --- Save one wide NetCDF per variable ---
print("Building and saving wide-format NetCDFs ...\n")

basin_coord = xr.DataArray(basin_ids, dims='basin_id', name='basin_id')
lat_coord   = xr.DataArray(lat_list,  dims='basin_id', name='latitude')
lon_coord   = xr.DataArray(lon_list,  dims='basin_id', name='longitude')

for var in VARIABLES:
    if not var_arrays[var]:
        print(f"Skipping {var} — no data found across any basin.")
        continue

    print(f"  Processing {var} ...")

    # Stack all basins along a new 'basin_id' dimension
    da_wide = xr.concat(var_arrays[var], dim=basin_coord)
    da_wide = da_wide.assign_coords(
        latitude=lat_coord,
        longitude=lon_coord
    )
    da_wide.name = var

    # Build output dataset with clean metadata
    ds_out = da_wide.to_dataset()
    ds_out.attrs = {
        'title':       f'AORC CAMELS forcing: {var}',
        'source':      f'HydroShare resource {HS_RESOURCE_ID}',
        'institution': 'University of Utah — johnsonrc-group1',
        'variable':    var,
        'n_basins':    len(basin_ids),
        'time_start':  str(da_wide.time.values[0]),
        'time_end':    str(da_wide.time.values[-1]),
        'note':        'Wide format: dims=(time, basin_id). Ready for LSTM training.',
    }

    out_path = processed_dir / f"AORC_CAMELS_{var}_1980_2024_hourly.nc"
    ds_out.to_netcdf(
        out_path,
        encoding={var: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}}
    )
    print(f"    Saved → {out_path}")

# =============================================================================
# STEP 4: SAVE BASIN METADATA CSV
# Useful for filtering basins, mapping gauge IDs to basin attributes, etc.
# =============================================================================

meta_df = pd.DataFrame({
    'basin_id': basin_ids,
    'latitude':  lat_list,
    'longitude': lon_list,
})
meta_path = processed_dir / 'basin_metadata.csv'
meta_df.to_csv(meta_path, index=False)
print(f"\nBasin metadata saved → {meta_path}")

# =============================================================================
# SUMMARY
# =============================================================================

print(f"""
{'='*60}
  DONE
  Processed files in: {processed_dir}

  Files produced:
    - AORC_CAMELS_<variable>_1980_2024_hourly.nc  (× {len(VARIABLES)})
        dims: (time, basin_id={len(basin_ids)})
        format: wide, float32, zlib compressed
    - basin_metadata.csv

  Next steps for LSTM:
    1. Download CAMELS streamflow observations:
       https://ral.ucar.edu/solutions/products/camels
    2. Align streamflow (daily) to forcing timestep (hourly or daily mean)
    3. Stack variables: shape → (time, basins, 8 features)
    4. Normalize per-variable across training period
    5. Train/val/test split (recommend: train<2000, val 2000-2010, test>2010)
{'='*60}
""")