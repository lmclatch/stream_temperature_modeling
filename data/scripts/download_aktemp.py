

import requests
import pandas as pd
import os
import time

work_dir = '/uufs/chpc.utah.edu/common/home/u1566670/civil-group1/McLatchy/alaska'  

# Change to that directory
os.chdir(work_dir)

BASE_URL = "https://u5ozso2814.execute-api.us-west-2.amazonaws.com/api/public"

def get_all_stations():
    """Get list of all stations"""
    url = f"{BASE_URL}/stations"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return []

def get_station_series(station_id):
    """Get all series for a specific station"""
    url = f"{BASE_URL}/stations/{station_id}/series"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return []

def download_series_data(series_id):
    """Download daily data for a series"""
    url = f"{BASE_URL}/series/{series_id}/daily"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    return None

# Build catalog
print("Building catalog...")
catalog = []
stations = get_all_stations()

# Initialize counters 
streams_count = 0
lakes_skipped = 0
for i, station in enumerate(stations, 1):
    station_id = station['id']
    
    # Handle None values properly
    waterbody_type = station.get('waterbody_type')
    if waterbody_type:
        waterbody_type = waterbody_type.upper()
    else:
        waterbody_type = None
    
    waterbody_name = station.get('waterbody_name', '')
    
    # Filter: only include STREAM, skip everything else (lakes, None, etc.)
    if waterbody_type != 'STREAM':
        lakes_skipped += 1
        continue
    
    streams_count += 1
    
    if i % 50 == 0:
        print(f"  Checked {i}/{len(stations)} stations... (kept {streams_count} streams, skipped {lakes_skipped} non-streams)")
    
    series_list = get_station_series(station_id)
    
    for series in series_list:
        catalog.append({
            'station_id': station_id,
            'station_code': station.get('code'),
            'provider_code': station.get('provider_code'),
            'waterbody_type': waterbody_type,
            'waterbody_name': waterbody_name if waterbody_name else 'UNKNOWN',
            'latitude': station.get('latitude'),
            'longitude': station.get('longitude'),
            'series_id': series['id'],
            'start_datetime': series['start_datetime'],
            'end_datetime': series['end_datetime'],
            'n_values': series['n_values'],
            'interval': series['interval'],
            'file_filename': series.get('file_filename')
        })
    
    time.sleep(0.1) # Sleep to avoid overwhelming the API

# Save catalog
catalog_df = pd.DataFrame(catalog)
catalog_df.to_csv('aktemp_catalog_streams_only.csv', index=False)

print(f"\n✓ Catalog complete!")
print(f"  Total stations checked: {len(stations)}")
print(f"  Stream stations: {catalog_df['station_id'].nunique()}")
print(f"  Total series (streams only): {len(catalog_df)}")
print(f"  Saved to: {os.path.join(work_dir, 'aktemp_catalog_streams_only.csv')}")

# Show summary
print("\nSeries per station:")
print(catalog_df.groupby('station_id').size().describe())

print("\nProviders:")
print(catalog_df['provider_code'].value_counts().head(10))

#Download full catalog data
# Step 1: Build a complete catalog with ALL waterbody types
print("Building complete catalog (all waterbody types)...")
catalog = []
stations = get_all_stations()

print(f"Total stations: {len(stations)}\n")

for i, station in enumerate(stations, 1):
    station_id = station['id']
    
    # Handle None values properly
    waterbody_type = station.get('waterbody_type')
    if waterbody_type:
        waterbody_type = waterbody_type.upper()
    else:
        waterbody_type = 'UNKNOWN'
    
    waterbody_name = station.get('waterbody_name', '')
    
    # NO FILTERING - include everything (streams, lakes, unknown)
    
    if i % 50 == 0:
        print(f"  Processed {i}/{len(stations)} stations...")
    
    series_list = get_station_series(station_id)
    
    for series in series_list:
        catalog.append({
            'station_id': station_id,
            'station_code': station.get('code'),
            'provider_code': station.get('provider_code'),
            'waterbody_type': waterbody_type,
            'waterbody_name': waterbody_name if waterbody_name else 'UNKNOWN',
            'latitude': station.get('latitude'),
            'longitude': station.get('longitude'),
            'placement': station.get('placement'),
            'series_id': series['id'],
            'start_datetime': series['start_datetime'],
            'end_datetime': series['end_datetime'],
            'n_values': series['n_values'],
            'interval': series['interval'],
            'file_filename': series.get('file_filename')
        })
    
    time.sleep(0.1)  # Be polite to the API

# Save complete catalog
catalog_df = pd.DataFrame(catalog)
catalog_df.to_csv('aktemp_catalog_complete.csv', index=False)

print(f"\n✓ Catalog complete!")
print(f"  Total stations: {len(stations)}")
print(f"  Total series: {len(catalog_df)}")

print(f"\nWaterbody types:")
print(catalog_df['waterbody_type'].value_counts())

print(f"\nSeries per station:")
print(catalog_df.groupby('station_id').size().describe())

print(f"\nTop providers:")
print(catalog_df['provider_code'].value_counts().head(10))

print(f"\nSaved to: {os.path.join(work_dir, 'aktemp_catalog_complete.csv')}")