import pandas as pd 
from pathlib import Path

''' Pull CAMELS attributes for the 90 stream temperature basins from the CAMELS attribute file, and save as a separate csv.
This is a simple filter of the full CAMELS attribute file '''

CAMELS = Path('/uufs/chpc.utah.edu/common/home/civil-group1/CAMELS')
BASIN_LIST = '/uufs/chpc.utah.edu/common/home/u1566670/hydroinformatics/stream_temperature_modeling/alaska/camels_temperature_availability_hourly.csv'
OUT_DIR = Path('/uufs/chpc.utah.edu/common/home/johnsonrc-group1/hourly_streamtemp/hourly/attributes')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / 'camels_attributes_90basins.csv'

#Pull basin IDS
df = pd.read_csv(BASIN_LIST, dtype={'site': str})
keep = df[(df['has_temp'] == True) & (df['n_years'] >= 4.0) & (df['pct_missing'] < 10.0)]
my_basins = sorted(keep['site'].str.zfill(8).unique())
print(f'Target basins: {len(my_basins)}')

# Load and merge all attribute files
files = ['camels_topo.txt', 'camels_clim.txt', 'camels_vege.txt',
         'camels_soil.txt', 'camels_geol.txt', 'camels_hydro.txt', 'camels_name.txt']

merged = None
for f in files:
    path = CAMELS / f
    a = pd.read_csv(path, sep=';', dtype={'gauge_id': str})
    a['gauge_id'] = a['gauge_id'].str.zfill(8)
    print(f'  {f}: {len(a)} basins, {len(a.columns)-1} attributes')
    if merged is None:
        merged = a
    else:
        overlap = [c for c in a.columns if c in merged.columns and c != 'gauge_id']
        a = a.drop(columns=overlap)
        merged = merged.merge(a, on='gauge_id', how='outer')

print(f'\nTotal CAMELS basins: {len(merged)}')
print(f'Total attributes: {len(merged.columns) - 1}')

sub = merged[merged['gauge_id'].isin(my_basins)].copy()
print(f'\nMatched: {len(sub)} / {len(my_basins)}')

missing = set(my_basins) - set(sub['gauge_id'])
if missing:
    print(f'Missing from CAMELS: {sorted(missing)}')

sub = sub.sort_values('gauge_id').reset_index(drop=True)
sub.to_csv(OUT, index=False)
print(f'\nWrote {len(sub.columns)-1} attributes for {len(sub)} basins')
print(f'  -> {OUT}')
