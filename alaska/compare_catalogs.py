import pandas as pd
import os
import requests
import time

work_dir = '/uufs/chpc.utah.edu/common/home/u1566670/civil-group1/McLatchy/alaska'
os.chdir(work_dir)

# Read the FULL AKTEMP catalog (not the streams-only one)
aktemp_catalog = pd.read_csv('aktemp_catalog_complete.csv')

print(f"AKTEMP catalog loaded: {len(aktemp_catalog)} series")
print(f"Unique stations: {aktemp_catalog['station_id'].nunique()}")
print(f"\nWaterbody types in catalog:")
print(aktemp_catalog['waterbody_type'].value_counts())

# Read your metadata file (handle the header comments)
def read_aktemp_metadata(filepath):
    """Read AKTEMP metadata file, skipping header comments"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the line with station_id header
    for i, line in enumerate(lines):
        if line.startswith('station_id'):
            data_start = i
            break
    
    # Read from that line onwards
    df = pd.read_csv(filepath, skiprows=data_start)
    return df

# Path to your metadata folder
metadata_folder = '/uufs/chpc.utah.edu/common/home/u1566670/civil-group1/McLatchy/alaska/station_metadata'

# List of your region files
region_files = [
    'AKTEMP-Arctic.csv',
    'AKTEMP-Juneau.csv',
    'AKTEMP-LowerYukon.csv',
    'AKTEMP-MiddleYukon.csv',
    'AKTEMP-Northwest.csv',
    'AKTEMP-Southcentral.csv',
    'AKTEMP-Southwest.csv'
]

print("\n" + "="*70)
print("READING REGIONAL METADATA FILES")
print("="*70)

# Read all metadata files
all_my_stations = []
for filename in region_files:
    filepath = os.path.join(metadata_folder, filename)
    
    if os.path.exists(filepath):
        print(f"\n✓ Reading: {filename}")
        df = read_aktemp_metadata(filepath)
        df['region'] = filename.replace('AKTEMP-', '').replace('.csv', '')
        print(f"  Stations: {len(df)}")
        all_my_stations.append(df)

# Concatenate all regions
my_stations_combined = pd.concat(all_my_stations, ignore_index=True)
print(f"\n{'='*70}")
print(f"✓ Combined metadata: {len(my_stations_combined)} total stations")
print(f"\nWaterbody types in your metadata:")
print(my_stations_combined['waterbody_type'].value_counts())

# Match by station CODE (not ID)
my_station_codes = set(my_stations_combined['code'].dropna())
catalog_station_codes = set(aktemp_catalog['station_code'].dropna())

print(f"\n" + "="*70)
print("STATION CODE COMPARISON")
print("="*70)
print(f"Your metadata: {len(my_station_codes)} unique station codes")
print(f"AKTEMP catalog: {len(catalog_station_codes)} unique station codes")

# Stations in both
in_both = my_station_codes & catalog_station_codes
print(f"\n✓ Station codes in BOTH: {len(in_both)}")

# In your metadata but not in catalog
in_mine_not_catalog = my_station_codes - catalog_station_codes
print(f"\n⚠ Station codes in YOUR metadata but NOT in catalog: {len(in_mine_not_catalog)}")

# In catalog but not in your metadata
in_catalog_not_mine = catalog_station_codes - my_station_codes
print(f"\nℹ Station codes in CATALOG but NOT in your metadata: {len(in_catalog_not_mine)}")

# Detailed comparison for matched stations
print(f"\n" + "="*70)
print("DETAILED COMPARISON (Matched Stations)")
print("="*70)

matched = my_stations_combined[my_stations_combined['code'].isin(catalog_station_codes)]
print(f"\nAnalyzing {len(matched)} matched stations...")

comparison = []
for _, my_station in matched.iterrows():
    station_code = my_station['code']
    
    # Get catalog info for this station CODE
    catalog_series = aktemp_catalog[aktemp_catalog['station_code'] == station_code]
    
    if len(catalog_series) > 0:
        comparison.append({
            'my_station_id': my_station['station_id'],
            'catalog_station_id': catalog_series.iloc[0]['station_id'],
            'station_code': station_code,
            'provider_code': my_station['provider_code'],
            'region': my_station['region'],
            'waterbody_name': my_station['waterbody_name'],
            'waterbody_type': my_station['waterbody_type'],
            'my_series_count': my_station['series_count'],
            'catalog_series_count': len(catalog_series),
            'series_match': my_station['series_count'] == len(catalog_series),
        })

comparison_df = pd.DataFrame(comparison)

if len(comparison_df) > 0:
    # Check for mismatches
    mismatches = comparison_df[~comparison_df['series_match']]
    if len(mismatches) > 0:
        print(f"\n⚠ Stations with different series counts: {len(mismatches)}")
        print("\nFirst 10 mismatches:")
        print(mismatches[['station_code', 'region', 'waterbody_type', 'my_series_count', 'catalog_series_count']].head(10).to_string(index=False))
    else:
        print("\n✓ All matched stations have correct series counts!")
    
    # Save comparison report
    comparison_df.to_csv('comparison_report.csv', index=False)
    print(f"\n✓ Detailed comparison saved to: comparison_report.csv")
    
    # Summary by region
    print(f"\n" + "="*70)
    print("SUMMARY BY REGION")
    print("="*70)
    region_summary = comparison_df.groupby('region').agg({
        'station_code': 'count'
    }).rename(columns={'station_code': 'matched_stations'})
    print(region_summary)
    
    # Summary by waterbody type
    print(f"\n" + "="*70)
    print("SUMMARY BY WATERBODY TYPE")
    print("="*70)
    waterbody_summary = comparison_df.groupby('waterbody_type').agg({
        'station_code': 'count'
    }).rename(columns={'station_code': 'matched_stations'})
    print(waterbody_summary)
    
    # Create download list using CATALOG station IDs
    matched_catalog_station_ids = comparison_df['catalog_station_id'].unique()
    download_list = aktemp_catalog[aktemp_catalog['station_id'].isin(matched_catalog_station_ids)]
    download_list.to_csv('series_to_download.csv', index=False)
    
    print(f"\n" + "="*70)
    print("DOWNLOAD LIST CREATED")
    print("="*70)
    print(f"✓ Series to download: {len(download_list)}")
    print(f"  Stations: {download_list['station_id'].nunique()}")
    print(f"  Total data points: {download_list['n_values'].sum():,}")
    print(f"  Estimated download size: ~{download_list['n_values'].sum() * 50 / 1024 / 1024:.1f} MB")
    print(f"  Saved to: series_to_download.csv")
else:
    print("\n✗ No matches found!")


# Load the full download list
download_list = pd.read_csv('series_to_download.csv')

print("Original download list:")
print(f"  Total series: {len(download_list)}")
print(f"\nBreakdown by waterbody type:")
print(download_list['waterbody_type'].value_counts())

# Filter to streams only
streams_only = download_list[download_list['waterbody_type'] == 'STREAM']

print(f"\n{'='*70}")
print("FILTERED TO STREAMS ONLY")
print(f"{'='*70}")
print(f"Series to download: {len(streams_only)}")
print(f"Stations: {streams_only['station_id'].nunique()}")
print(f"Total data points: {streams_only['n_values'].sum():,}")
print(f"Estimated size: ~{streams_only['n_values'].sum() * 50 / 1024 / 1024:.1f} MB")

# Save filtered list
streams_only.to_csv('series_to_download_streams_only.csv', index=False)
print(f"\nSaved to: series_to_download_streams_only.csv")

# Show breakdown by region
print(f"\nStreams by region:")
print(streams_only.groupby('provider_code')['series_id'].count().sort_values(ascending=False).head(10))


BASE_URL = "https://u5ozso2814.execute-api.us-west-2.amazonaws.com/api/public"

def download_series_data(series_id):
    """Download daily data for a series"""
    url = f"{BASE_URL}/series/{series_id}/daily"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    return None

# Load the STREAMS ONLY download list
download_list = pd.read_csv('series_to_download_streams_only.csv')

print("="*70)
print("AKTEMP DATA DOWNLOAD - STREAMS ONLY")
print("="*70)
print(f"Series to download: {len(download_list)}")
print(f"Stations: {download_list['station_id'].nunique()}")
print(f"Estimated size: ~{download_list['n_values'].sum() * 50 / 1024 / 1024:.1f} MB")
print(f"Estimated time: ~{len(download_list) * 0.5 / 60:.0f} minutes")
print("="*70)

proceed = input("\nProceed with download? (y/n): ")
if proceed.lower() != 'y':
    print("Download cancelled.")
    exit()

# Create output directory
output_dir = 'aktemp_downloaded_data_streams'
os.makedirs(output_dir, exist_ok=True)

# Track progress
successful = 0
failed = 0
failed_series = []

print(f"\nDownloading to: {output_dir}")
print("Starting download...\n")

for idx, row in download_list.iterrows():
    series_id = row['series_id']
    station_id = row['station_id']
    station_code = row['station_code']
    
    print(f"[{idx+1}/{len(download_list)}] Series {series_id} ({station_code})...", end=' ')
    
    df = download_series_data(series_id)
    
    if df is not None and len(df) > 0:
        # Add metadata columns
        df['station_id'] = station_id
        df['station_code'] = station_code
        df['series_id'] = series_id
        
        # Save
        filename = f"{output_dir}/series_{series_id}_{station_code}.csv"
        df.to_csv(filename, index=False)
        
        print(f"✓ ({len(df)} rows)")
        successful += 1
    else:
        print("✗ FAILED")
        failed += 1
        failed_series.append({'series_id': series_id, 'station_code': station_code})
    
    time.sleep(0.5)  # Be polite to the API
    
    # Progress update every 100 series
    if (idx + 1) % 100 == 0:
        print(f"\n--- Progress: {idx+1}/{len(download_list)} ({successful} successful, {failed} failed) ---\n")

print(f"\n{'='*70}")
print("DOWNLOAD COMPLETE!")
print(f"{'='*70}")
print(f"Successful: {successful}")
print(f"Failed: {failed}")
print(f"Output directory: {output_dir}")

if failed > 0:
    failed_df = pd.DataFrame(failed_series)
    failed_df.to_csv('failed_downloads.csv', index=False)
    print(f"\nFailed series saved to: failed_downloads.csv")