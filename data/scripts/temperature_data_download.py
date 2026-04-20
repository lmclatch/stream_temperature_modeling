import pandas as pd
import dataretrieval.nwis as nwis
import os

#Filter for availability of data (at least 2 years of data with less than 10% missing)
df = pd.read_csv('/uufs/chpc.utah.edu/common/home/u1566670/hydroinformatics/stream_temperature_modeling/alaska/camels_temperature_availability_hourly.csv', dtype={'site': str})

# fix n_years units
df['n_years'] = df['n_years'] / (35040 / 365)

# apply filters
filtered = df[
    (df['has_temp'] == True) &
    (df['n_years'] >= 2.0) &        # at least 2 full years
    (df['pct_missing'] < 10.0)      # less than 10% missing
]

SITE_IDS = filtered['site'].str.zfill(8).tolist()
print(f"Sites passing filter: {len(SITE_IDS)}")
print(SITE_IDS)

# parameter codes
# 00010 = water temperature (°C)
PARAMS = ['00010']
 
OUTPUT_DIR = '/uufs/chpc.utah.edu/common/home/johnsonrc-group1/CAMELS_US/hourly/stream_temperature'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_site(site_id, start='2000-01-01', end='2023-12-31'):
    print(f'Downloading {site_id}...')
    try:
        df, meta = nwis.get_iv(
            sites=site_id,
            parameterCd=PARAMS,
            start=start,
            end=end
        )
        # resample from 15-min to hourly
        df.index = pd.to_datetime(df.index, errors='coerce')
        df.index = df.index.tz_localize(None)
        print(df.head())

         # remove qualifier/code columns
        df.drop(columns=[c for c in df.columns if c.endswith('_cd')], inplace=True, errors='ignore')

        # find the real temperature column
        #Some columns end with _2, some do not
        temp_cols = [c for c in df.columns if c.startswith('00010')]
        if not temp_cols:
            raise ValueError(f'No temperature column found. Columns: {df.columns.tolist()}')
        temp_col = temp_cols[0]
        # convert to numeric
        df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')

        # resample to hourly
        df_hourly = df[temp_col].resample('1h').mean()
        out_path = os.path.join(OUTPUT_DIR, f'{site_id}.csv')
        df_hourly.to_csv(out_path)
        print(f'  Saved {len(df_hourly)} hourly records → {out_path}')
        
    except Exception as e:
        print(f'  FAILED {site_id}: {e}')

if __name__ == '__main__':
    for site in SITE_IDS:
        download_site(site)
