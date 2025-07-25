import pandas as pd
from pathlib import Path
import sys
from time import time
import matplotlib.pyplot as plt
sys.path.append(str(Path.cwd().parents[1]))

from logger import logger
from utilities import get_region_border, set_catch_ratio
from plot_data import plot_all_catches, extract_copernicus_variable
from fetch_data import fetch_data_from_fiskeridirektoratet, fetch_data_from_copernicus_marine


REGIONS_PATH = '/home/anna/msc_oppgave/fish-forecast/areas.json'
MAC_REGIONS = ["FA_JunNov_download"]
MAC_TIMES = {"FA_JunNov_download": ['01-01', '12-31']}

years = [2011, 2024] # start and end year for the data to inspect

variables = ["so"]

depths_to_keep = [
    4.940250e-01, 1.541375e+00, 2.645669e+00, 3.819495e+00, 5.078224e+00,
    6.440614e+00, 7.929560e+00, 9.572997e+00, 1.140500e+01, 1.346714e+01,
    1.581007e+01, 1.849556e+01, 2.159882e+01, 2.521141e+01, 2.944473e+01,
    3.443415e+01, 4.034405e+01, 4.737369e+01, 5.576429e+01, 6.580727e+01,
    7.785385e+01, 9.232607e+01, 1.097293e+02, 1.306660e+02, 1.558507e+02,
    1.861256e+02, 2.224752e+02]
depths = [depths_to_keep[0], depths_to_keep[-1]]  # first and last depth level in the dataset

region_coords = get_region_border(REGIONS_PATH, MAC_REGIONS[0])

dataset_dict = {
    "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    "longitude": [region_coords["SW"]["lon"], region_coords["SE"]["lon"]],
    "latitude": [region_coords["SW"]["lat"], region_coords["NE"]["lat"]],
    "time": [f"2021-06-30T00:00:00", f"2021-06-30T00:00:00"],
    "variables": ["so"],
    "depths": [depths[0], depths[0]]
}

df = pd.read_csv("/home/anna/msc_oppgave/fish-forecast/Data-Exploration/existing_areas/data_download_area_2011_2024.csv", sep=";")
# df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
# df['Location'] = (df['CatchLat'], df['CatchLon'])


start_time = time()
test_data = fetch_data_from_copernicus_marine(dataset_dict)
logger.info(f"Data fetched in {time() - start_time:.2f} seconds")

start_time = time()
mat = extract_copernicus_variable(test_data, variables, depth=depths[0])
logger.info(f"Variable extracted in {time() - start_time:.2f} seconds")


for variable in variables:
    logger.info(f"samples in _area: {mat[variable].shape}")
    plt.figure()
    plt.title(f"{variable} at depth {depths[0]} m")
    plt.imshow(mat[variable])
    plt.colorbar(label=variable)

logger.info("Plotting complete")
plt.show()

