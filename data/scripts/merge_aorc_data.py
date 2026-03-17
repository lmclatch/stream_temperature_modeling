"""
Merges per-water-year AORC CSVs into one continuous CSV per basin.

Input:  01013500_1980_to_2024_agg_rounded_WR1980.csv  (× 45 years × 671 basins)
Output: 01013500_hourly_aorc.csv                      (× 671 basins)

Run on CHPC interactive node or as a SLURM job.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

aorc_dir = Path('/uufs/chpc.utah.edu/common/home/johnsonrc-group1/CAMELS_US/hourly/aorc_hourly')

# Get all unique basin IDs from filenames
all_files  = sorted(aorc_dir.glob('*_agg_rounded_WR*.csv'))
basin_ids  = sorted(set(f.name.split('_')[0] for f in all_files))

print(f"Basins found : {len(basin_ids)}")
print(f"Total files  : {len(all_files)}")
print(f"Years/basin  : {len(all_files) // len(basin_ids)}")
print()

failed = []
for basin_id in tqdm(basin_ids, desc="Merging basins", unit="basin"):
    try:
        # Get all water year files for this basin, sorted chronologically
        basin_files = sorted(aorc_dir.glob(f"{basin_id}_*_agg_rounded_WR*.csv"))

        # Read and concatenate all years
        df = pd.concat(
            [pd.read_csv(f, parse_dates=['time']) for f in basin_files],
            ignore_index=True
        )

        # Sort by time and drop any duplicates at water year boundaries
        df = df.sort_values('time').drop_duplicates(subset='time').reset_index(drop=True)

        # Save as single file in NeuralHydrology format
        out_path = aorc_dir / f"{basin_id}_hourly_aorc.csv"
        df.to_csv(out_path, index=False)

    except Exception as e:
        failed.append((basin_id, str(e)))

# Remove the original per-year files to save space
print("\nRemoving per-year files ...")
for f in all_files:
    f.unlink()

print(f"""
{'='*60}
  MERGE COMPLETE
{'='*60}
  Basins merged : {len(basin_ids) - len(failed)}
  Output files  : {len(list(aorc_dir.glob('*_hourly_aorc.csv')))}
  Location      : {aorc_dir}
""")

if failed:
    print(f"  WARNING: {len(failed)} basins failed:")
    for basin_id, err in failed:
        print(f"    {basin_id}: {err}")