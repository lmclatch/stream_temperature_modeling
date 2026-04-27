# save as: 01_download_nwis_discharge.py
"""
Pull hourly discharge (00060) from NWIS for the 90 filtered CAMELS basins.
Temperature already exists at AORC_StreamTemp/hourly/stream_temperature/.
"""
import pandas as pd
import dataretrieval.nwis as nwis
from pathlib import Path

BASIN_LIST = "/uufs/chpc.utah.edu/common/home/u1566670/hydroinformatics/stream_temperature_modeling/conus/camels_temperature_availability_hourly.csv"
OUTPUT_DIR = Path("/uufs/chpc.utah.edu/common/home/johnsonrc-group1/hourly_streamtemp/hourly/streamflow")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASIN_LIST, dtype={'site': str})
filtered = df[
    (df['has_temp'] == True) &
    (df['n_years'] >= 4.0) &
    (df['pct_missing'] < 10.0)
]
SITE_IDS = filtered['site'].str.zfill(8).tolist()
print(f"Sites to download: {len(SITE_IDS)}")

START = '1980-01-01'
END = '2025-12-31'

def download_site(site_id):
    out_path = OUTPUT_DIR / f'{site_id}.csv'
    if out_path.exists():
        print(f'  {site_id} already exists, skipping.')
        return

    print(f'Downloading {site_id}...')
    try:
        df_raw, meta = nwis.get_iv(
            sites=site_id,
            parameterCd='00060',
            start=START,
            end=END
        )
        if len(df_raw) == 0:
            print(f'  No Q data for {site_id}')
            return

        df_raw.index = pd.to_datetime(df_raw.index, errors='coerce')
        df_raw.index = df_raw.index.tz_localize(None)

        df_raw.drop(columns=[c for c in df_raw.columns if c.endswith('_cd')],
                    inplace=True, errors='ignore')

        q_cols = [c for c in df_raw.columns if c.startswith('00060')]
        if not q_cols:
            print(f'  No 00060 column for {site_id}: {df_raw.columns.tolist()}')
            return

        q = pd.to_numeric(df_raw[q_cols[0]], errors='coerce')
        q_hourly = q.resample('1h').mean()
        q_hourly.name = 'discharge_cfs'
        q_hourly.to_csv(out_path)
        print(f'  Saved {len(q_hourly)} hourly records ({q_hourly.notna().sum()} non-null) → {out_path.name}')

    except Exception as e:
        print(f'  FAILED {site_id}: {e}')

if __name__ == '__main__':
    for i, site in enumerate(SITE_IDS, 1):
        print(f'[{i}/{len(SITE_IDS)}]', end=' ')
        download_site(site)
    print('Done.')