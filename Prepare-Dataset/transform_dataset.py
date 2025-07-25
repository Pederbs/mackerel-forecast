"""
Script for transforming saved data using Gaussian anamorphic transformation.
This script computes empirical CDFs for specified variables and depths,
applies Gaussian anamorphosis, and saves the transformed dataset.
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from logger import logger
from typing import Optional
from scipy.stats import norm
from utilities import get_region_border

def compute_empirical_cdfs(
    ds: xr.Dataset,
    variables: list,
    depths: list,
    sample_size: int = 10_000
) -> dict:
    """
    For each var, flatten values across time/lat/lon,
    draw up to sample_size points, sort -> empirical CDF sample.
    Returns: { var: { depth: sorted_sample_array | None } }
    """
    ds_sub = ds[variables].sel(depth=depths, method='nearest')
    cdfs: dict[str, dict[float, Optional[np.ndarray]]] = {}
    for var in tqdm(variables, desc="Sampling empirical CDFs"):
        cdfs[var] = {}
        for depth in depths:
            all_vals = ds_sub[var].sel(depth=depth, method='nearest').values.ravel()
            valid = all_vals[~np.isnan(all_vals)]
            if valid.size == 0:
                cdfs[var][depth] = None
            else:
                # Draw subsample
                n = min(sample_size, valid.size)
                idx = np.random.choice(valid.size, size=n, replace=False)
                samp = valid[idx]
                # Sort to get CDF
                cdfs[var][depth] = np.sort(samp)
    return cdfs


def apply_gaussian_anamorphosis(
    ds: xr.Dataset,
    cdfs: dict,
    variables: list,
    depths: list,
    mu: float = 0.5,
    sigma: float = 1/6,
    show_progress: bool = True
) -> xr.Dataset:
    """
    Given precomputed cdfs[var][depths], map every point to Gaussian(mu,sigma).
    Returns a new Dataset same shape as ds[variables].sel(depths).
    """
    ds_sub = ds[variables].sel(depth=depths, method='nearest')
    out    = xr.Dataset(coords=ds_sub.coords)

    for var in tqdm(variables, desc="Applying Gaussian anamorphosis", disable=not show_progress):
        out[var] = xr.full_like(ds_sub[var], np.nan, dtype=np.float32)
        for depth in depths:
            sample = cdfs.get(var, {}).get(depth)
            if sample is None:
                continue

            # build uniform‐quantile grid for interpolation
            m = sample.size
            # avoid 0 and 1 so norm.ppf stays finite
            u = (np.arange(1, m+1) / (m+1)).astype(np.float32)

            da    = ds_sub[var].sel(depth=depth, method='nearest')
            vals  = da.values         # shape (time,lat,lon)
            valid = np.isfinite(vals)

            # vectorized CDF interpolation
            # map any x<min(sample)→u[0], x>max(sample)→u[-1]
            q = np.interp(vals, sample, u, left=u[0], right=u[-1])

            # uniform → Gaussian
            z = norm.ppf(q) * sigma + mu
            z[~valid] = np.nan

            out[var].loc[{"depth": depth}] = z.astype(np.float32)

    return out

def transform_per_day(
    ds: xr.Dataset,
    cdfs: dict,
    variables: list,
    depths: list,
    mu: float = 0.5,
    sigma: float = 1/6
) -> xr.Dataset:
    """
    Slice ds into 1-day chunks, apply the same full period anamorphosis (cdfs),
    then concatenate back into one time series.
    """
    # floor every timestamp to its date
    dates = np.unique(ds.time.dt.floor("D").values)
    out_slices = []
    for date in tqdm(dates, desc="Per-day transform"):
        # take that 24h window
        window = ds.sel(time=slice(date, date + np.timedelta64(1, "D")))
        transformed = apply_gaussian_anamorphosis(
            window, cdfs, variables, depths, mu, sigma, show_progress=False
        )
        out_slices.append(transformed)
    # recombine along time
    return xr.concat(out_slices, dim="time")

def normalize_variables(ds: xr.Dataset, variables: list, depths: list, method: str='min-max') -> xr.Dataset:
    """
    Normalize specified variables and depths in the dataset to have zero mean and unit variance.
    """
    implemented_methods = ['min-max']
    if method == 'min-max':
        for var in tqdm(variables, desc="Normalizing variables"):
            if var in ds:
                for depth in depths:
                    da = ds[var].sel(depth=depth)
                    da_min = da.min(dim="time", skipna=True).compute()
                    da_max = da.max(dim="time", skipna=True).compute()
                    ds[var].loc[{"depth": depth}] = (da - da_min) / (da_max - da_min)
    else:
        raise ValueError("Unsupported normalization method. use one of: " + ", ".join(implemented_methods))

    return ds

def extract_area_of_interest(ds: xr.Dataset, region_coord: dict) -> xr.Dataset:
    """
    Extract a specific area of interest from the dataset based on latitude and longitude ranges.
    """
    lat_range = (region_coord['SW']['lat'], region_coord['NE']['lat'])
    lon_range = (region_coord['SW']['lon'], region_coord['NE']['lon'])

    return ds.where(
        (ds.latitude >= lat_range[0]) & (ds.latitude <= lat_range[1]) &
        (ds.longitude >= lon_range[0]) & (ds.longitude <= lon_range[1]),
        drop=True
    )


if __name__ == "__main__":
    #---------------------------------------------------------------------------------------
    # define paths and parameters
    #---------------------------------------------------------------------------------------
    logger.info("Starting transformation of datasets")
    catch_path = '/home/anna/msc_oppgave/fish-forecast/catch_ERS_VMS_FINAL_DATASET.csv'
    phy_path = '/home/anna/msc_oppgave/fish-forecast/Data/mac_winter_areas/phy/'
    phy_no_catch_path = '/mnt/otherdrive/mac_winter_areas/no_catch_dates/phy/'
    # For data before 2011 use resampled_d3
    bio_path = '/home/anna/msc_oppgave/fish-forecast/Data/mac_winter_areas/bio_resampled/'
    bio_no_catch_path = '/mnt/otherdrive/mac_winter_areas/no_catch_dates/bio_resampled/'

    # Save paths for the new dataset
    ds_path = '/home/anna/msc_oppgave/fish-forecast/Data/all_catch_small/'
    ds_new_name = 'GAT'  # 'anamorphosed'

    # phy_depths = [4.940250e-01]
    # bio_depths = [3.0]
    # bio_variables = ["chl", "kd", "no3", "nppv", "o2", "phyc", "po4", "si", "zooc"]
    # phy_variables = ["so", "thetao", "uo", "vo"]
    bio_variables = ["no3","o2", "zooc"]
    phy_variables = ["thetao", "uo"]

    phy_depths = [1.5413750410079956, 29.444730758666992]
    bio_depths = [22.0, 29.0]
    # bio_depths = [3.0] # can only use depth 3

    SAMPLES = 50_000

    start_year = '2019'
    end_year = '2024'
    start_date = '07-01'
    end_date =   '11-30'
    
    region_name = 'test'
    region_border = get_region_border('/home/anna/msc_oppgave/fish-forecast/areas.json',
                                      region_name)

    shape = []  
    df = pd.read_csv(catch_path, sep=';')
    shape.append(df.shape[0])
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    # Filter the DataFrame for the specified date range
    df = df[(df['Date'].dt.month >= 7) & (df['Date'].dt.month <= 11)]
    df = df[(df['Date'].dt.year >= int(start_year)) & (df['Date'].dt.year <= int(end_year))]

    # Filter catches to only include those within the region
    df_catch = df[df['Catch'] == True]
    df_catch_region = df_catch[
        (df_catch['CatchLat'] >= region_border['SW']['lat']) & 
        (df_catch['CatchLat'] <= region_border['NE']['lat']) &
        (df_catch['CatchLon'] >= region_border['SW']['lon']) & 
        (df_catch['CatchLon'] <= region_border['NE']['lon'])
    ]

    # Get no-catch days (these don't have lat/lon constraints)
    df_no_catch = df[df['Catch'] == False]

    # Combine filtered catches with all no-catch days
    # df_filtered = pd.concat([df_catch_region, df_no_catch], ignore_index=True) # uncomment for balanced dataset
    df_filtered = df_catch_region

    # Sort by date to maintain chronological order
    df_filtered = df_filtered.sort_values('Date').reset_index(drop=True)

    shape.append(df_filtered.shape[0])
    logger.info(f"Filtered out {shape[0] - shape[1]} rows, remaining: {shape[1]}")
    logger.info(f"Catch records in region: {len(df_catch_region)}")
    logger.info(f"No-catch records: {len(df_no_catch)}")

    # Get dates of interest from the filtered DataFrame
    dates_of_interest = df_filtered['Date'].dt.strftime('%Y-%m-%d').unique() 
    logger.debug(f'Using {len(dates_of_interest)} dates')

    
    # Save the dates of interest in the dataset path
    if not os.path.exists(ds_path):
        os.makedirs(ds_path)
    logger.info(f"Saving dates of interest to {ds_path}dates_of_interest.csv")
    df_filtered.to_csv(f"{ds_path}dates_of_interest.csv", index=False, sep=';')


    # exit()
    #---------------------------------------------------------------------------------------
    # Find files and prepare paths
    #---------------------------------------------------------------------------------------

    # Create output paths for both new datasets
    bio_output = os.path.join(f'{ds_path}bio/', ds_new_name)
    phy_output = os.path.join(f'{ds_path}phy/', ds_new_name)

    # get sorted list of files
    bio_files = sorted(glob.glob(f"{bio_path}/*.nc"))
    phy_files = sorted(glob.glob(f"{phy_path}/*.nc"))
    
    # add no-catch paths if they exist
    if os.path.exists(phy_no_catch_path):
        logger.info(f"Adding no-catch PHY files from {phy_no_catch_path}")
        phy_files += sorted(glob.glob(f"{phy_no_catch_path}/*.nc"))
    if os.path.exists(bio_no_catch_path):
        logger.info(f"Adding no-catch BIO files from {bio_no_catch_path}")
        bio_files += sorted(glob.glob(f"{bio_no_catch_path}/*.nc"))


    bio_files.sort()
    phy_files.sort()

    # Only keep files that are in dates of interest
    bio_files = [f for f in bio_files if any(date in f for date in dates_of_interest)]
    phy_files = [f for f in phy_files if any(date in f for date in dates_of_interest)]
   
    logger.debug(f"Bio files: {len(bio_files)}")
    logger.debug(f"Phy files: {len(phy_files)}")

    first_date = bio_files[0].split('/')[-1][:-3]
    last_date = bio_files[-1].split('/')[-1][:-3]
    logger.info(f"Processing data from {first_date} to {last_date}")


    #----------------------------------------------------------------------------------------
    # Open Biochemical dataset and transform
    #----------------------------------------------------------------------------------------
    logger.info("Opening BIO datasets")
    bio_ds = xr.open_mfdataset(bio_files, combine='by_coords')
    bio_ds_region = extract_area_of_interest(bio_ds, region_border)

    logger.info("Transforming biochemical data")
    cdfs = compute_empirical_cdfs(bio_ds_region, 
                                  bio_variables, 
                                  bio_depths, 
                                  sample_size=SAMPLES)
    
    ds_transformed = apply_gaussian_anamorphosis(bio_ds_region, 
                                                 cdfs,
                                                 bio_variables, 
                                                 bio_depths,
                                                 mu=0.5,
                                                 sigma=1/6)

    # ds_transformed = normalize_variables(bio_ds_region,
    #                                      bio_variables,
    #                                      bio_depths,
    #                                      method='min-max')

    os.makedirs(bio_output, exist_ok=True)
    bio_save_path = f"{bio_output}/{first_date}_{last_date}.nc"
    logger.info(f"Saving transformed BIO: {bio_save_path}")
    ds_transformed.to_netcdf(bio_save_path)

    del bio_ds_region, ds_transformed
    #delete cdfs if defined
    if 'cdfs' in locals():
        del cdfs
    #----------------------------------------------------------------------------------------
    # Open physical datasets and transform
    #----------------------------------------------------------------------------------------
    logger.info("Opening PHY datasets")
    phy_ds = xr.open_mfdataset(phy_files, combine='by_coords')
    phy_ds = phy_ds.sel(depth=phy_depths,
                        method="nearest",
                        tolerance=1e-2)
    phy_ds_region = extract_area_of_interest(phy_ds, region_border)

    logger.info("Transforming physical data")
    cdfs = compute_empirical_cdfs(phy_ds_region, 
                              phy_variables, 
                              phy_depths, 
                              sample_size=SAMPLES)

    phy_transformed = apply_gaussian_anamorphosis(phy_ds_region, 
                                                 cdfs,
                                                 phy_variables, 
                                                 phy_depths,
                                                 mu=0.5,
                                                 sigma=1/6)

    # phy_transformed = normalize_variables(phy_ds_region,
    #                                      phy_variables,
    #                                      phy_depths,
    #                                      method='min-max')

    os.makedirs(phy_output, exist_ok=True)
    phy_save_path = f"{phy_output}/{first_date}_{last_date}.nc"
    logger.info(f"Saving transformed PHY: {phy_save_path}")
    phy_transformed.to_netcdf(phy_save_path)

