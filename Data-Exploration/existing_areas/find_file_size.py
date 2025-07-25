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

variables = ["so", "thetao", "uo", "vo"]
depth = 4.940250e-01    # first depth level in the dataset

depths_to_keep = [
    4.940250e-01, 1.541375e+00, 2.645669e+00, 3.819495e+00, 5.078224e+00,
    6.440614e+00, 7.929560e+00, 9.572997e+00, 1.140500e+01, 1.346714e+01,
    1.581007e+01, 1.849556e+01, 2.159882e+01, 2.521141e+01, 2.944473e+01,
    3.443415e+01, 4.034405e+01, 4.737369e+01, 5.576429e+01, 6.580727e+01,
    7.785385e+01, 9.232607e+01, 1.097293e+02, 1.306660e+02, 1.558507e+02,
    1.861256e+02, 2.224752e+02]
depths = [depths_to_keep[0], depths_to_keep[14]]  # first and last depth level in the dataset
print(f"Using depths: {depths}")


region_coords = get_region_border(REGIONS_PATH, MAC_REGIONS[0])

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


# Physical data
logger.info("Fetching physical data...")
start_time = time()
phy_data = fetch_data_from_copernicus_marine(phy_dict)
logger.info(f"Data fetched in {time() - start_time:.2f} seconds")

logger.warning(f"Data size: {phy_data.nbytes / 1_000_000} MB")
logger.info(phy_data)


# Bio Small data
logger.info("Fetching physical data...")
start_time = time()
bio_s_data = fetch_data_from_copernicus_marine(bio_dict_small)
logger.info(f"Data fetched in {time() - start_time:.2f} seconds")

logger.warning(f"Data size: {bio_s_data.nbytes / 1_000_000} MB")
logger.info(bio_s_data)


# Bio Big data
logger.info("Fetching physical data...")
start_time = time()
bio_b_data = fetch_data_from_copernicus_marine(bio_dict_big)
logger.info(f"Data fetched in {time() - start_time:.2f} seconds")

logger.warning(f"Data size: {bio_b_data.nbytes / 1_000_000} MB")
logger.info(bio_b_data)

# def extract_copernicus_variable(ds, variable="temperature", depth: float = 0):
#     """
#     Extract a 2D NumPy array at the nearest depth and first time step.
#     Returns a dict: var_name -> np.ndarray of shape (lat, lon).
#     """
#     import numpy as np
#     result = {}
#     vars_ = [variable] if isinstance(variable, str) else variable

#     for var in vars_:
#         if var not in ds:
#             continue

#         # 1) select nearest depth, 2) drop time dim, 3) get raw numpy array
#         da = ds[var].sel(depth=depth, method="nearest").isel(time=0)
#         arr = da.values.astype(np.float32)      # now a pure np.ndarray
#         arr = np.flipud(arr)                    # flip north-up

#         result[var] = arr

#     return result

if True:
    plot_var = [phy_dict["variables"][0], bio_dict_small["variables"][0], bio_dict_big["variables"][1]]
    mat = dict()
    for variable, ds, depth in zip(plot_var, [phy_data, bio_s_data, bio_b_data], [4.940250e-01, 2.0, 3.0]):
        logger.info(f"Extracting variable {variable} at depth {depth} m")
        result = extract_copernicus_variable(ds, variable, depth=depth)
        mat[variable] = result[variable]
        plt.figure()
        plt.title(f"{variable} at depth {depth} m")
        plt.imshow(mat[variable])
    # breakpoint()
    plt.show()

