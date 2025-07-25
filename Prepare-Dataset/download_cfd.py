"""
Script to download data files (.nc) from Copernicus Marine. Given a .csv file with dates, it fetches the data for each date and saves it in a specified directory.
"""

import pandas as pd
from pathlib import Path
import sys
from time import time
import matplotlib.pyplot as plt

from logger import logger
from utilities import get_region_border, set_catch_ratio, plot_date_distribution
from plot_data import plot_all_catches, extract_copernicus_variable
from fetch_data import fetch_data_from_fiskeridirektoratet, fetch_data_from_copernicus_marine


REGIONS_PATH = '/home/anna/msc_oppgave/fish-forecast/areas.json'
MAC_REGIONS = ["FA_JunNov_download"]
MAC_TIMES = {"FA_JunNov_download": ['06-01', '11-30']}

depths = [4.940250e-01, 2.944473e+01]  # first and last depth level in found from phy dataset
years = [2011, 2024] # start and end year for the data to inspect
catch_ratio = 0.5  # ratio of catch to all samples in the dataset

region_coords = get_region_border(REGIONS_PATH, MAC_REGIONS[0])

csv_path  = "/home/anna/msc_oppgave/fish-forecast/download_area_2011_2024_06-11_57p_filtered.csv"
if Path(csv_path).exists():
    df = pd.read_csv(csv_path, sep=";")
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Location'] = list(zip(df['CatchLat'], df['CatchLon']))
    df_full = df.copy()
else:
    df_full = pd.DataFrame()
    for y in range(years[0], years[1] + 1):
        test_params_fdir = {
                    "dataset_path": "/home/anna/msc_oppgave/data/fiskeridirektoratet/",
                    "time": [f"{y}-{MAC_TIMES[MAC_REGIONS[0]][0]}", f"{y}-{MAC_TIMES[MAC_REGIONS[0]][1]}"],
                    "species": "MAC"
                }

        test_params_fdir.update(region_coords)
        ds = fetch_data_from_fiskeridirektoratet(test_params_fdir, day_samples=False)

        ds = set_catch_ratio(test_params_fdir, ds, catch_ratio, method='continuous')

        df_full = pd.concat([df_full, ds])

    df_full = df_full.reset_index(drop=True)
    percent = int(catch_ratio * 100)    
    df_full.to_csv(f"download_area_{years[0]}_{years[1]}_06-11_{percent}p.csv", sep=';', index=False)

# only download dates without catch
df_full = df_full[~df_full['Catch']]


all_dates = set(df_full['Date'].unique())

# dates_wocatch = set(df_full[df_full['CatchLat'].isna()]['Date'].unique())
# dates_wcatch = set(df_full[~df_full['CatchLat'].isna()]['Date'].unique())

# logger.info(f"catch: {len(dates_wcatch)}, no catch: {len(dates_wocatch)}, total: {len(all_dates)}")
# logger.info(f"Ratio: {len(dates_wcatch) / len(all_dates):.2f} catch to all dates")
# plot_date_distribution(dates_wcatch, dates_wocatch, MAC_TIMES[MAC_REGIONS[0]][0], MAC_TIMES[MAC_REGIONS[0]][1])


SAVE_PATH = "/mnt/otherdrive/mac_winter_areas/no_catch_dates/"

phy_dict = {
    "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    "longitude": [region_coords["SW"]["lon"], region_coords["SE"]["lon"]],
    "latitude": [region_coords["SW"]["lat"], region_coords["NE"]["lat"]],
    "time": [f"2021-06-30T00:00:00", f"2021-06-30T00:00:00"],
    "variables": ["so", "thetao", "uo", "vo"],
    "depths": depths
}
bio_dict_small = phy_dict.copy()
bio_dict_small["dataset_id"] = "cmems_mod_arc_bgc_anfc_ecosmo_P1D-m"
bio_dict_small["variables"] = ["chl", "kd", "model_depth", "no3", "nppv", "o2", "phyc", "po4", "si", "zooc"]

bio_dict_big = bio_dict_small.copy()
bio_dict_big["dataset_id"] = "cmems_mod_arc_bgc_my_ecosmo_P1D-m"

# Create directories if they do not exist
Path(SAVE_PATH + "phy").mkdir(parents=True, exist_ok=True)
Path(SAVE_PATH + "bio").mkdir(parents=True, exist_ok=True)

# sort all_dates from big to small
all_dates = sorted(all_dates, reverse=True)



# Download data from Copernicus Marine
for date in all_dates:
    date_str = date.strftime("%Y-%m-%dT%H:%M:%S")
    phy_dict["time"] = [date_str, date_str]
    name_str = date.strftime("%Y-%m-%d")

    # skip if the file already exists
    if Path(SAVE_PATH + f"phy/{name_str}.nc").exists() and Path(SAVE_PATH + f"bio/{name_str}.nc").exists():
        logger.info(f"Data for {name_str} already exists, skipping...")
        continue

    if date >= pd.to_datetime("2021-07-01"):
        phy_dict["dataset_id"] = "cmems_mod_glo_phy_myint_0.083deg_P1D-m"
    else:
        phy_dict["dataset_id"] = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
    # Physical data
    logger.info(f"Fetching physical data for {date_str}...")
    phy_data = fetch_data_from_copernicus_marine(phy_dict)
    
    phy_data.to_netcdf(SAVE_PATH + f"phy/{name_str}.nc")

    # Bio Big data
    if date < pd.to_datetime("2019-01-01"):
        bio_dict_big["time"] = [date_str, date_str]
        bio_dict_big["depths"] = [3.0, 3.0]
        logger.info(f"Fetching bio data (big) for {date_str}...")
        bio_b_data = fetch_data_from_copernicus_marine(bio_dict_big)
        
        bio_b_data.to_netcdf(SAVE_PATH + f"bio/{name_str}.nc") 

    # Bio Small data   
    else:
        bio_dict_small["time"] = [date_str, date_str]
        logger.info(f"Fetching bio data (small) for {date_str}...")
        bio_s_data = fetch_data_from_copernicus_marine(bio_dict_small)
        
        bio_s_data.to_netcdf(SAVE_PATH + f"bio/{name_str}.nc")
        # write to a .txt file the dates in bio small data
        with open(SAVE_PATH + "bio/bio_small_dates.txt", "a") as f:
            f.write(f"{date.strftime('%Y-%m-%d')}\n")


    
