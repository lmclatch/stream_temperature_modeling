import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import s3fs
import warnings
from shapely.geometry import box
from datetime import datetime
from exactextract import exact_extract
from aorc_utils import get_aggregation_code, display_shapefile_map, da_animate

'''
Adapted from:
Salehabadi, H., D. Tarboton, A. Nassar, A. M. Castronova, P. Dash, A. Patel, F. Baig (2026). 
Jupyter Notebooks for the Retrieval of AORC Data for Hydrologic Analysis, HydroShare,
 http://www.hydroshare.org/resource/72ea9726187e43d7b50a624f2acf591f
'''

#Goal: download AORC data for CONUS and Alaska
#Basic Parameters
# Start date - In Year-Month-Day format the earliest start date can be '1979-02-01'
start_datetime = '2020-01-01' 

# End date - In Year-Month-Day format the latest end date can be '2023-01-31'
end_datetime = '2020-12-31'

# File path to the shapefile
shapefile_path = ".shp"

# Basin name (this will be used on the maps and plots)
basin_name = "Great Salt Lake Basin"

# Variable name to retrieve data (look at the following table for valid variable names)
variable_name = 'APCP_surface'

# User-defined aggregation interval - valid values are 'hour','day','month','year'
agg_interval = 'hour'

# Read the shapefile
gdf = gpd.read_file(shapefile_path)

## Create a list of years to retrieve data 
start_yr = datetime.strptime(start_datetime, '%Y-%m-%d').year
end_yr = datetime.strptime(end_datetime, '%Y-%m-%d').year
yrs = list(range(start_yr, end_yr+1))

## Loading data (AORC data are organized by years, look at https://noaa-nws-aorc-v1-1-1km.s3.amazonaws.com/index.html)
# Base URL
base_url = f's3://noaa-nws-aorc-v1-1-1km'
# Creating a connection to Amazon S3 bucket using the s3fs library (https://s3fs.readthedocs.io/en/latest/api.html).
s3_out = s3fs.S3FileSystem(anon=True)              # access S3 as if it were a file system. 
fileset = [s3fs.S3Map(                             # maps each year's Zarr dataset from S3 to a local-like object.
            root=f"s3://{base_url}/{yr}.zarr",     # Zarr dataset for each year
            s3=s3_out,                             # connection
            check=False                            # checking if the dataset exists before trying to load it
        ) for yr in yrs]                           # loops through each year

## Load data for specified years and veriable of interest using the xarray library
ds_yrs = xr.open_mfdataset(fileset, engine='zarr')
da_yrs_var = ds_yrs[variable_name]
variable_long_name = da_yrs_var.attrs.get('long_name')
da_yrs_var

'''
Use the following to check the Coordinate Reference System (CRS) of the 
shapefile and reproject it if it does not match the dataset CRS. 
The AORC data uses EPSG:4326 coordinate system, which corresponds to the World Geodetic System 1984 
(WGS84) geographic latitude and longitude reference system'''

data_crs = da_yrs_var.rio.crs
if gdf.crs == data_crs:
    print(f"CRS of Data and Shapefile --->  {data_crs}")

# Reproject shapefile if needed
if gdf.crs != data_crs:
    print(f'Original Shapefile CRS --->  {gdf.crs}')
    gdf = gdf.to_crs(data_crs)
    print(f'CRS of Data and Reprojected Shapefile  --->  {data_crs}')

#Temporal aggregation:
## Aggregation time interval
agg_code = get_aggregation_code(agg_interval)

## Retrive units
units = da_yrs_var.attrs.get('units', 'No units specified')

## Retrieve data for the bounding box first for efficiency, then clip the data for the basin.
bounding_box = box(*gdf.total_bounds)
x_min, y_min, x_max, y_max = bounding_box.bounds
da_bbox = da_yrs_var.sel(latitude=slice(y_min, y_max), longitude=slice(x_min, x_max))

if variable_name == 'APCP_surface':
    units = f"mm/{agg_interval}"
    # Temporal aggregation
    da_bbox_TimeAgg = da_bbox.loc[dict(time=slice(start_datetime, end_datetime))].resample(time=agg_code).sum()

else:
    # Temporal aggregation
    da_bbox_TimeAgg = da_bbox.loc[dict(time=slice(start_datetime, end_datetime))].resample(time=agg_code).mean()

# Ensure data has CRS for exact_extract
da_bbox_TimeAgg.rio.write_crs(gdf.crs, inplace=True)

# Clip data for the entire basin
geom_union = gdf.unary_union
da_over_area = da_bbox_TimeAgg.rio.clip([geom_union], gdf.crs, all_touched=True)
# Note: The `all_touched` parameter controls whether to include all cells that are even partially touched by the vector (all_touched = True) or only those whose 
# center falls inside the vector (all_touched = False). Set all_touched = True to ensure that every cell that intersects the vector, even if only slightly, is 
# included. We will then use exactextract library to extract the exact zonal statistics. 

da_over_area

#Spatial aggregation:
%%time

# Suppress a warning for potential future changes in GDAL (Geospatial Data Abstraction Library).
warnings.filterwarnings("ignore") 

# Compute mean for each subbasins using exactextract
df_zonal_agg_subbasins = exact_extract(da_over_area, gdf, "mean", include_cols=["name"], output='pandas')

# The following lines will organaize the output dataframe to make it easier to analyze and plot,
# and also calculate the mean for the entire basin based on the means of the subbasins.

df_mean_over_area = df_zonal_agg_subbasins.copy()

# Format time labels
time_coords = pd.to_datetime(da_over_area['time'].values)
formatted_dates = time_coords.strftime(
    '%Y-%m-%d %H:%M:%S' if agg_interval == 'hour' else  
    '%Y-%m-%d' if agg_interval == 'day' else
    '%Y-%m' if agg_interval == 'month' else
    '%Y'
)

# Rename the columns to match time periods
df_mean_over_area.columns = ['subbasin'] + list(formatted_dates)  

# Set "subbasin" as index for easier plotting
df_mean_over_area.set_index('subbasin', inplace=True)

# Transpose data for plotting (time as index, subbasins as columns)
df_mean_over_area = df_mean_over_area.T 

# Calculate mean values for the entire basin based on subbasins. This will proceed only if there are more than one subbasin. 
if len(gdf) > 1: # Check if there are more than one subbasin in the gdf
    subbasin_area = gdf_projected['Projected_Area_km2']
    # Multiply the values by the respective subbasin areas
    weighted_values = df_mean_over_area.multiply(subbasin_area.values, axis=1)
    # Calculate the weighted mean for the entire basin
    weighted_mean_entire_basin = weighted_values.sum(axis=1) / total_area
    # Add the "Entire Basin" column
    df_mean_over_area['Entire Basin'] = weighted_mean_entire_basin

# Print the units
print(f"units --> {units}")
# Display the result
df_mean_over_area