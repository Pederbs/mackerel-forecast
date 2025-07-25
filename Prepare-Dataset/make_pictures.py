"""
Script to make datasets for fish forecasting by making use of various data sources and plotting functions.
"""

from plot_data import plot_copernicus_variable, plot_norkyst800_temperature, plot_norkyst800_salinity, plot_daily_catches, extract_copernicus_variable
import os
import xarray as xr
from utilities import get_region_border
from logger import logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load or create the regions dictionary.
region_name = "test"
region_coords = get_region_border("areas.json", region_name, square=False)

dataset = 'all_catch_small'
data_base_path = f'/home/anna/msc_oppgave/fish-forecast/Data/{dataset}'

csv_path = f'/home/anna/msc_oppgave/fish-forecast/TEST_MAC_ERS_VMS_COMPLETE.csv'
phy_path = f'{data_base_path}/phy/GAT/'
bio_path = f'{data_base_path}/bio/GAT/'
output_path = f'/home/anna/msc_oppgave/fish-forecast/Datasets/{dataset}'

# depths_phy = [4.940250e-01]
# depths_bio = [3]
# bio_variables = ["chl", "kd", "no3", "nppv", "o2", "phyc", "po4", "si", "zooc"]
# phy_variables = ["thetao", "so", "vo", "uo"]

bio_variables = ["no3","o2", "zooc"]
phy_variables = ["thetao", "uo"]

depths_phy = [1.5413750410079956, 29.444730758666992]
depths_bio = [22.0, 29.0]
# depths_bio = [3.0] # can only use depth 3


# Remove and recreate output directories for bio variables
for variable in bio_variables:
    for depth in depths_bio:
        path = f"{output_path}/{variable}/{depth}"

        # Remove existing files in the directory
        if os.path.exists(path):
            for file in os.listdir(path):
                os.remove(os.path.join(path, file))
        # Create directory
        os.makedirs(path, exist_ok=True)

# Remove and recreate output directories for phy variables
for variable in phy_variables:
    for depth in depths_phy:
        depth_rounded = round(depth, 2)  # Round depth to 2 decimal places
        depth_str = str(depth_rounded).replace('.', '_')  # Replace '.' with '_' for directory names
        path = f"{output_path}/{variable}/{depth_str}"

        # Remove existing files in the directory
        if os.path.exists(path):
            for file in os.listdir(path):
                os.remove(os.path.join(path, file))
        # Create directory
        os.makedirs(path, exist_ok=True)


catch_df = pd.read_csv(f'{data_base_path}/dates_of_interest.csv', sep=';')
# # filter catch_df to only include catches within the region
# lat_range = (region_coords['SW']['lat'], region_coords['NE']['lat'])
# lon_range = (region_coords['SW']['lon'], region_coords['NE']['lon'])

# catch_df = catch_df[(catch_df['CatchLat'] >= lat_range[0]) & (catch_df['CatchLat'] <= lat_range[1]) &
#                     (catch_df['CatchLon'] >= lon_range[0]) & (catch_df['CatchLon'] <= lon_range[1])]

dates = catch_df['Date'].unique()
# Plot all bio files
bio_ds = xr.open_mfdataset(os.path.join(bio_path, '*.nc'), combine='by_coords')
for date in tqdm(dates, desc="Plotting bio data"):
    bio_ds_date = bio_ds.sel(time=date, method='nearest')
    for depth in depths_bio:
        mat = extract_copernicus_variable(bio_ds_date, bio_variables, depth=depth, keep_nan=False)
        for variable in bio_variables:
            im = mat[variable]
            plt.imsave(f"{output_path}/{variable}/{depth}/{date}.png", im, cmap='gray', vmin=0, vmax=1)
logger.info("Finished plotting bio data.")

# Plot all phy files
phy_ds = xr.open_mfdataset(os.path.join(phy_path, '*.nc'), combine='by_coords')
for date in tqdm(dates, desc="Plotting phy data"):
    phy_ds_date = phy_ds.sel(time=date, method='nearest')
    for depth in depths_phy:
        depth_rounded = round(depth, 2)  # Round depth to 2 decimal places
        depth_str = str(depth_rounded).replace('.', '_')  # Replace '.' with '_' for directory names
        mat = extract_copernicus_variable(phy_ds_date, phy_variables, depth=depth, keep_nan=False)
        for variable in phy_variables:
            im = mat[variable]
            plt.imsave(f"{output_path}/{variable}/{depth_str}/{date}.png", im, cmap='gray', vmin=0, vmax=1)
logger.info("Finished plotting phy data.")


# Make plot of catches:
ers_vms = pd.read_csv(csv_path, sep=';')

df = pd.concat([catch_df, ers_vms])

path = f'{output_path}/catch/'
# Remove existing files in the directory
if os.path.exists(path):
    for file in os.listdir(path):
        os.remove(os.path.join(path, file))
# Create directory
os.makedirs(path, exist_ok=True)

plot_daily_catches(df, region_coords, path, use_catch_weight=False)
logger.info("Finished plotting catch data.")

logger.info("All data has been plotted and saved to the output directory.")