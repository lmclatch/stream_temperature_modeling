#!/usr/bin/env python
"""
De-accumulate ERA5-Land daily-cumulative fields in CAMELSH-derived basin files.

ERA5-Land convention: total_precipitation, surface_net_solar_radiation,
surface_net_thermal_radiation, and potential_evaporation are accumulated from
00 UTC each day, with the daily reset such that the value at hour 01 represents
the first fresh hourly increment.

Per-hour increment = value(H) - value(H-1), except at the reset point (h01)
where the raw value IS the increment.
"""

import xarray as xr
import numpy as np
from pathlib import Path
import sys

TS_DIR = Path('/uufs/chpc.utah.edu/common/home/johnsonrc-group1/CAMELS_hourly/era5_90basins/time_series')
OUT_DIR = Path('/uufs/chpc.utah.edu/common/home/johnsonrc-group1/CAMELS_hourly/era5_90basins/time_series_deaccumulated')
OUT_DIR.mkdir(exist_ok=True)

# ERA5-Land daily-accumulating variables
# Non-negative by physics: precip, net solar radiation
# Can be negative: net thermal radiation (downward minus upward, often negative at night),
#                  potential evaporation (sign convention varies; ERA5-Land uses negative for evap loss)
ACCUM_VARS_NONNEG = ['total_precipitation', 'surface_net_solar_radiation']
ACCUM_VARS_SIGNED = ['surface_net_thermal_radiation', 'potential_evaporation']
ALL_ACCUM_VARS = ACCUM_VARS_NONNEG + ACCUM_VARS_SIGNED

def deaccumulate(arr, hours, clip_negative=True):
    """De-accumulate ERA5-Land daily-cumulative field, reset at hour 01 UTC.
    
    arr:   1D numpy array of cumulative values
    hours: 1D numpy array of UTC hour for each timestep (0-23)
    clip_negative: if True, set negative increments to zero (use only for non-negative-by-physics fields)
    """
    deacc = np.empty_like(arr)
    deacc[0] = arr[0]  # edge case for very first timestep
    
    # Vectorized: diff first, then overwrite reset hours with raw values
    diffs = np.diff(arr)              # length N-1
    deacc[1:] = diffs
    reset_mask = (hours[1:] == 1)     # at h01 the raw value IS the increment
    deacc[1:][reset_mask] = arr[1:][reset_mask]
    
    if clip_negative:
        # Tiny negatives can occur from float precision in non-negative fields
        deacc = np.where(deacc < 0, 0, deacc)
    
    return deacc

def process_file(in_path, out_path, verbose=True):
    ds = xr.open_dataset(in_path)
    hours = ds.time.dt.hour.values
    
    ds_fixed = ds.copy(deep=True)
    
    for v in ACCUM_VARS_NONNEG:
        if v in ds.data_vars:
            arr = ds[v].values
            new_arr = deaccumulate(arr, hours, clip_negative=True)
            ds_fixed[v] = (ds[v].dims, new_arr)
            ds_fixed[v].attrs = dict(ds[v].attrs)
    
    for v in ACCUM_VARS_SIGNED:
        if v in ds.data_vars:
            arr = ds[v].values
            new_arr = deaccumulate(arr, hours, clip_negative=False)
            ds_fixed[v] = (ds[v].dims, new_arr)
            ds_fixed[v].attrs = dict(ds[v].attrs)
    
    # Document what was done
    ds_fixed.attrs['processing_note'] = (
        'De-accumulated daily-cumulative ERA5-Land variables: '
        f'{", ".join(ALL_ACCUM_VARS)}. '
        'Original CAMELSH values were daily-cumulative with reset at hour 01 UTC '
        '(the raw GEE ERA5-Land convention). Per-hour increments computed as '
        'diff between consecutive timesteps, with reset hours preserving raw value.'
    )
    ds_fixed.attrs['processing_date'] = str(np.datetime64('now'))
    
    ds_fixed.to_netcdf(out_path)
    
    if verbose:
        # Quick sanity check
        if 'total_precipitation' in ds.data_vars:
            orig_annual = ds['total_precipitation'].values.sum() / 45 * 1000
            new_annual = ds_fixed['total_precipitation'].values.sum() / 45 * 1000
            print(f'  {in_path.name}: precip {orig_annual:.0f} -> {new_annual:.0f} mm/yr')
    
    ds.close()
    ds_fixed.close()

def main():
    files = sorted(TS_DIR.glob('*.nc'))
    print(f'Found {len(files)} files in {TS_DIR}')
    print(f'Output dir: {OUT_DIR}')
    print(f'De-accumulating: {ALL_ACCUM_VARS}')
    print()
    
    # Test on first file
    print('=== Test on first file ===')
    process_file(files[0], OUT_DIR / files[0].name, verbose=True)
    print()
    
    # Pause for user to verify
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        print('=== Processing remaining files ===')
        for f in files[1:]:
            process_file(f, OUT_DIR / f.name, verbose=True)
        print(f'\nDone. {len(files)} files written to {OUT_DIR}')
    else:
        print('Test complete. Inspect the output, then re-run with --all to process all 90 files:')
        print(f'  python {sys.argv[0]} --all')

if __name__ == '__main__':
    main()