import pandas as pd
import dataretrieval.nwis as nwis
import os

# Load availability CSV
df_avail = pd.read_csv(
    '/uufs/chpc.utah.edu/common/home/u1566670/hydroinformatics/stream_temperature_modeling/conus/camels_temperature_availability_hourly.csv',
    dtype={'site': str}
)

# Apply the locked 4-year filter (matches the 90-basin set)
filtered = df_avail[
    (df_avail['has_temp'] == True) &
    (df_avail['n_years'] >= 4.0) &
    (df_avail['pct_missing'] < 10.0)
]

SITE_IDS = filtered['site'].str.zfill(8).tolist()
print(f"Sites passing filter: {len(SITE_IDS)}")

PARAMS = ['00010']   # water temperature (°C)
OUTPUT_DIR = '/uufs/chpc.utah.edu/common/home/johnsonrc-group1/hourly_streamtemp/hourly/stream_temperature'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_site(site_id, start='1980-01-01', end='2025-12-31'):
    out_path = os.path.join(OUTPUT_DIR, f'{site_id}.csv')
    if os.path.exists(out_path):
        print(f'  {site_id} already exists, skipping')
        return

    print(f'Downloading {site_id}...')
    try:
        df_raw, meta = nwis.get_iv(
            sites=site_id,
            parameterCd=PARAMS,
            start=start,
            end=end
        )
        if len(df_raw) == 0:
            print(f'  No data for {site_id}')
            return

        df_raw.index = pd.to_datetime(df_raw.index, errors='coerce')
        df_raw.index = df_raw.index.tz_localize(None)

        # remove qualifier/code columns
        df_raw.drop(columns=[c for c in df_raw.columns if c.endswith('_cd')],
                    inplace=True, errors='ignore')

        # find temperature column (may have _2, _3 suffix across basins)
        temp_cols = [c for c in df_raw.columns if c.startswith('00010')]
        if not temp_cols:
            raise ValueError(f'No temperature column found. Columns: {df_raw.columns.tolist()}')
        temp_col = temp_cols[0]

        df_raw[temp_col] = pd.to_numeric(df_raw[temp_col], errors='coerce')

        df_hourly = df_raw[temp_col].resample('1h').mean()
        df_hourly.name = 'temperature_C'
        df_hourly.to_csv(out_path)

        print(f'  Saved {len(df_hourly)} hourly records '
              f'({df_hourly.notna().sum()} non-null) → {site_id}.csv')

    except Exception as e:
        print(f'  FAILED {site_id}: {e}')


if __name__ == '__main__':
    for i, site in enumerate(SITE_IDS, 1):
        print(f'[{i}/{len(SITE_IDS)}]', end=' ')
        download_site(site)
    print('Done.')