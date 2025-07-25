"""
Utility functions for data processing.
"""

import os
import json
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from logger import logger
from fetch_data import fetch_data_from_copernicus_marine

from tqdm import tqdm


def get_region_border(regions_file: str, region_name: str, square: bool = False) -> dict:
    """
    Load the regions dictionary from a JSON file, and return the region coordinates.

    Arguments:
        regions_file (str): Path to the regions JSON file.

    Returns:
        dict: Dictionary with the region coordinates.
    """
    if not os.path.exists(regions_file):
        raise FileNotFoundError(f"Regions file '{regions_file}' not found.")

    with open(regions_file, "r") as f:
        regions =  json.load(f)
    if square is True:
        region = square_region_border(regions[region_name])
    else:
        region = regions[region_name]
    return region
    
def square_region_border(region: dict) -> dict:
    """
    Convert non square regions to sqaure regions
    """
    min_lon = region["SW"]["lon"]
    max_lon = region["NE"]["lon"]
    min_lat = region["SW"]["lat"]
    max_lat = region["NE"]["lat"]

    # Calculate the center of the rectangle.
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    # Determine the differences in both directions and choose the smallest.
    lon_diff = max_lon - min_lon
    lat_diff = max_lat - min_lat
    side = min(lon_diff, lat_diff)

    # Calculate the boundaries of the square.
    region["SW"]["lon"] = center_lon - side / 2
    region["NE"]["lon"] = center_lon + side / 2
    region["SW"]["lat"] = center_lat - side / 2
    region["NE"]["lat"] = center_lat + side / 2

    return region


def fetch_or_load_data(params):
    """
    DEPRICATED???\n
    Check if the data file for data from copernicus exists based on the parameters in the params dict.
    If the file exists, load it; otherwise, fetch the dataset and save it.
    """
    region_name   = params["region_name"]
    region_coords = params["region_coords"]
    time_range    = params["time_range"]
    fetch_vars    = params["fetch_variables"]

    # Create an output filename that encodes the region, time range, and variables.
    variables_str = "_".join(fetch_vars)
    filename = f"data/sst_{region_name}_{time_range[0]}_to_{time_range[1]}_{variables_str}.nc"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if os.path.exists(filename):
        logger.info(f"Data file '{filename}' already exists. Skipping download.")
        dataset = xr.open_dataset(filename)
    else:
        # logger.info(f"Fetching data for region '{region_name}' with coordinates: {region_coords}")
        data_request = {
            "dataset_id": "cmems_mod_arc_bgc_anfc_ecosmo_P1D-m",  # 2019 -> now
            # 'dataset_id' : 'cmems_mod_arc_bgc_my_ecosmo_P1D-m',   # 2007 -> 2019
            "longitude": [region_coords["SW"]["lon"], region_coords["NE"]["lon"]],
            "latitude": [region_coords["SW"]["lat"], region_coords["NE"]["lat"]],
            "time": time_range,
            "variables": fetch_vars
        }
        dataset = fetch_data_from_copernicus_marine(data_request)
        # logger.info(f"Data fetched successfully. Saving to '{filename}'")
        # dataset.to_netcdf(filename)
        # logger.info(f"Data saved to '{filename}'.")
    return dataset


def set_catch_ratio(param: dict, df: pd.DataFrame, desired_ratio: float, method: str) -> pd.DataFrame:
    """
    Calculate the ratio of days with catch/all dates in the period and adjust accordingly

    Parameters
    ----------
        param : dict
            Dictionary containing the start and end date. time : [start_date, end_date] date format: YYYY-MM-DD
        df : pd.DataFrame
            Dataframe containing the catch dates
        desired_ratio : float
            Desired ratio of catch days / no catch days (0.0-1.0)
        method : str
            Method to use for adding no-catch dates (e.g., "continuous" or leave blank for random)

    Returns:
    ----------
        pd.DataFrame: 
            A DataFrame with the adjusted catch dates.

    """
    # TODO: make the dates between catch dates picked first, currently seeing padding before and after the main catching dates...
    
    # Assuming that the dataframe only contains dates with catch
    df["Date"] = pd.to_datetime(df["Date"])

    catch_dates = df["Date"].unique()
    n_catch_dates = len(catch_dates)
    catch_ratio_df = 1.0 # TODO: change this if functionality for parsing df w/o catch is added

    # Start and end date extraction (YYYY-MM-DD)
    start_date = param["time"][0]
    end_date = param["time"][1]

    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    df["Date"] = pd.to_datetime(df["Date"])
    ratio = n_catch_dates / len(dates)
    logger.info(f"Period used (%): {n_catch_dates}/{len(dates)} = {ratio * 100:.2f}% -> Days w/o: {len(dates) - len(catch_dates)}")

    # If catches are extracted from fiskeridirektoratet, it will contain 100% catch dates
    # can it contain duplicates??
    if catch_ratio_df > desired_ratio: # Current is 1.
        no_catch_dates = list(set(dates) - set(catch_dates))
        # Calculate the number of dates to add to the dataframe.
        num_dates_to_add = int(n_catch_dates / desired_ratio - n_catch_dates)
        logger.debug(f"Want to add {num_dates_to_add} days from {len(no_catch_dates)} possible.")

        if method == 'continuous':
            throw, new_dates_set = generate_continuous_no_catch_dates(df, num_dates_to_add, start_date, end_date)
            new_dates = sorted(new_dates_set)
        elif len(no_catch_dates) < num_dates_to_add:
            # TODO: Implement widening of the time range to fulfill the desired ratio. (flag as argument to function, 
            # return updated params dict...) 
            logger.warning(
                f"Number of dates to add exceeds the number of dates without catch, "\
                f"adding all {len(no_catch_dates)} dates without catch. If this is not enough, "\
                f"please increase the time range."
                )
            new_dates = np.random.choice(no_catch_dates, len(no_catch_dates), replace=False)
        else:
            # Add random dates within the range to the dataframe.
            logger.debug(f"Adding {num_dates_to_add} dates to the dataframe.")
            new_dates = np.random.choice(no_catch_dates, num_dates_to_add, replace=False)
        new_dates = pd.DataFrame({
            "Date": new_dates, 
            "Rundvekt": np.nan,
            "CatchLat": np.nan,
            "CatchLon": np.nan,
            "Location": np.nan
        })
        # logger.info(f"New dates: {new_dates}")
        df = pd.concat([df, new_dates], ignore_index=True)
    
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")
    logger.info(
        f"Dates with catch (%): {n_catch_dates / len(df['Date'].unique()) * 100:.2f}"\
        f"-> Goal: {desired_ratio * 100:.2f} -> {len(df['Date'].unique())} days"
                )
        # df.to_csv("sample.csv", index=False, sep=";")

    return df

def generate_continuous_no_catch_dates(df: pd.DataFrame, n_dates : int, start_date: str, end_date: str) -> tuple:
    """
    Generate a set of continuous no-catch dates for a given DataFrame and time range.

    Needs more fixing? check winter area 50% catch

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the catch dates.
    start_date : str
        Start date of the time range (YYYY-MM-DD).
    end_date : str
        End date of the time range (YYYY-MM-DD).

    Returns
    -------
    tuple
        A tuple containing two sets:
        - catch_dates: Set of dates with catches.
        - no_catch_dates: Set of dates without catches.
    """
    no_catch_dates = set()

    catch_dates = set(df["Date"])
    # catch_dt = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in catch_dates]
    catch_dt = list(catch_dates)
    # find year (all dates should be in the same year)

    year = catch_dt[0].year

    # Compute midpoint for possible padding heuristic
    free_span = catch_dt[-1] - catch_dt[0]
    middle    = (catch_dt[0] + free_span / 2)
    middle    = pd.to_datetime(middle).normalize().to_pydatetime()

    logger.debug(f"Midpoint = {middle}, N = {n_dates}")

    # Full season and strip out actual catch days
    start = datetime.datetime(year, int(start_date[5:7]), int(start_date[8:]))
    end   = datetime.datetime(year, int(end_date[5:7]), int(end_date[8:]))
    season_dates = set(pd.date_range(start, end).to_pydatetime())
    season_no_catch = sorted(season_dates - set(catch_dt))

    # Pick up to N dates after the very first catch
    after_first = [d for d in season_no_catch if d > catch_dt[0]]
    pick_after  = after_first[:n_dates]
    no_catch_dates.update(pick_after)

    # If we need more dates, pad from the rest of the season_no_catch
    if len(pick_after) < n_dates:
        need = n_dates - len(pick_after)

        # Build the pool of remaining dates
        remaining_pool = [d for d in season_no_catch if d not in pick_after]

        # TODO: choose your padding strategy here. E.g.:
        #   A) closest to the midpoint:
        remaining_pool.sort(key=lambda d: abs((d - middle).days))

        #   B) or just the earliest in the season:
        # remaining_pool = sorted(remaining_pool)

        pad = remaining_pool[:need]
        no_catch_dates.update(pad)

    # plot_date_distribution expects strings, so:
    no_catch_strs = {d.strftime('%Y-%m-%d') for d in no_catch_dates}

    return catch_dates, no_catch_strs


def plot_date_distribution(catch_dates: set, no_catch_dates: set, start_date: str = '', end_date: str = '') -> None:
    """
    Plot the distribution of catch and no-catch dates over the years.
    
    Parameters
    ----------
    catch_dates : set
        Set of dates (str) with catches.
    no_catch_dates : set
        Set of dates (str) without catches.
    start_date : str
        First date to include in the plot (MM-DD).
    end_date : str
        Last date to include in the plot (MM-DD).
    """
    # Pre-parse your date strings once
    # catch_dt    = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in catch_dates]
    # no_catch_dt = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in no_catch_dates]
    catch_dt = catch_dates
    no_catch_dt = no_catch_dates
    min_year = min([d.year for d in catch_dt])
    max_year = max([d.year for d in catch_dt])
    years = list(range(min_year, max_year + 1))
    
    if start_date == '' or end_date == '':
        logger.warning("No start/end date provided, defining default start and end dates.")
        start_date = "01-01"
        end_date = "12-31"

    # Create a blank figure
    fig = plt.figure(figsize=(12, 2 * len(years)))

    for idx, year in enumerate(years):
        # Create the idx+1’th subplot in a len(years)×1 grid
        ax = fig.add_subplot(len(years), 1, idx + 1)

        # Define season start/end as datetimes
        start = datetime.datetime(year, int(start_date[:2]), int(start_date[3:]))
        end   = datetime.datetime(year, int(end_date[:2]), int(end_date[3:]))

        # Filter for this year’s season
        yc = [d for d in catch_dt    if start <= d <= end]
        yn = [d for d in no_catch_dt if start <= d <= end]

        # Plot blue bars for catches, red for no-catches
        w = 0.85
        colors = plt.cm.viridis(np.linspace(0, 1, 8))  # Use a colormap for colors
        ax.bar(yc, [1] * len(yc), color=colors[5], width=w)
        ax.bar(yn, [1] * len(yn), color=colors[6], width=w)

        # Limit x to the season, hide y-axis
        ax.set_xlim(start, end)
        ax.set_ylim(0, 1.5)
        ax.get_yaxis().set_visible(False)

        # Title each subplot by year
        ax.set_title(str(year))

        # Only the top subplot needs a legend
        # if idx == 0:
        ax.legend(['Catch', 'No Catch'], loc='upper right')

        ax.grid(axis='x', linestyle='--', alpha=0.5)
    # Beautify the x-axis dates and layout
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

def find_format(year):
    if year < 2021:
        return "%d.%m.%Y  %H:%M:%S" 
    elif year == 2021:
        return "%d.%m.%Y %H:%M:%S" 
    elif year == 2022:
        return "%d.%m.%Y %H.%M.%S,%f"
    elif year == 2023:
        return "%d.%m.%Y %H:%M:%S"
    elif year == 2024:
        return "%d-%b-%y %H.%M.%S.%f"


def find_vms_records(df_ers, df_vms, year):
    """
    Find VMS records for vessels that have reported catches in the ERS data
    """
    # VMS timestamps are in the example format 01-JAN-24 15.40.00.000000000
    # If the year is before 2022
    # date_format = find_format(year)
    df_vms['Tidspunkt (UTC)'] = pd.to_datetime(
        df_vms['Tidspunkt (UTC)'],
        format='mixed',
        dayfirst=True
    )

    df_vms_subset = df_vms[df_vms['Fartøyidentifikasjon']
                           .isin(df_ers['Fartøyidentifikasjon'].dropna())]
    print(f"Number of VMS records: {df_vms_subset.shape[0]}")

    # filter out locations that are not in the time range of the ERS data
    df_ers['Starttidspunkt'] = pd.to_datetime(df_ers['Starttidspunkt'], format='mixed', dayfirst=True)
    df_ers['Stopptidspunkt'] = pd.to_datetime(df_ers['Stopptidspunkt'], format='mixed', dayfirst=True)

    # df_ers['Starttidspunkt'].dt.strftime("%Y-%m-%d %H:%M:%S")
    # df_ers['Stopptidspunkt'].dt.strftime("%Y-%m-%d %H:%M:%S")
    # df_vms['Tidspunkt (UTC)'].dt.strftime("%Y-%m-%d %H:%M:%S")



    # Filter out timestamps that are outside the range of the ERS data
    start = df_ers['Starttidspunkt'].min()
    end   = df_ers['Stopptidspunkt'].max()
    mask = (
        (df_vms_subset['Tidspunkt (UTC)'] >= start) &
        (df_vms_subset['Tidspunkt (UTC)'] <= end)
    )
    
    df_vms_subset = df_vms_subset.loc[mask]
    print(f"VMS date: {df_vms_subset['Tidspunkt (UTC)'].min()} -> {df_vms_subset['Tidspunkt (UTC)'].max()}")
    print(f"Number of VMS records after initial time-filter:  {len(df_vms_subset)}")


    # loop over vessels in the ERS data and filter VMS records to only include locations reported in between the start and end time of the ERS data
    df_valid_vessels = pd.DataFrame()
    for vessel in tqdm(df_ers['Fartøyidentifikasjon'].unique(), desc="Filtering VMS records by ERS vessels"):
        ers_vessel_mask = df_ers['Fartøyidentifikasjon'] == vessel
        # for each row in the ERS data, get the start and end time
        for row in df_ers[ers_vessel_mask].itertuples():
            start = row.Starttidspunkt
            end = row.Stopptidspunkt

            # filter VMS records for the current vessel
            mask = (
                (df_vms_subset['Fartøyidentifikasjon'] == vessel) &
                (df_vms_subset['Tidspunkt (UTC)'] >= start) &
                (df_vms_subset['Tidspunkt (UTC)'] <= end)
            )
            valid_times = df_vms_subset.loc[mask]

            # Append the valid VMS records for the current vessel to the df_valid_vessels DataFrame
            df_valid_vessels = pd.concat([df_valid_vessels, valid_times])
    print(f"Number of VMS records after filtering by ERS vessels: {df_valid_vessels.shape[0]}")

    print(str(len(df_ers['Fartøyidentifikasjon'].unique())) + " unique vessels")
    print(str(len(df_valid_vessels['Fartøyidentifikasjon'].unique())) + " unique vessels")
    print(f"Missing VMS vessels: {set(df_ers['Fartøyidentifikasjon'].unique()) - set(df_valid_vessels['Fartøyidentifikasjon'].unique())}")

    # create an empty DataFrame to store the valid vessels
    df_return = pd.DataFrame()
    df_return['Starttidspunkt'] = df_valid_vessels['Tidspunkt (UTC)']
    df_return['Fartøyidentifikasjon'] = df_valid_vessels['Fartøyidentifikasjon']
    df_return['Source'] = df_valid_vessels['Source']

    if df_valid_vessels['Breddegrad'].dtype == 'float64':
        # If already float, just assign
        df_return['CatchLat'] = df_valid_vessels['Breddegrad']
        df_return['CatchLon'] = df_valid_vessels['Lengdegrad']
    else:
        # Because 2022 is an object for some reason
        # Convert to string, replace, then to float
        df_return['CatchLat'] = df_valid_vessels['Breddegrad'].astype(str).str.replace(',', '.').astype(float)
        df_return['CatchLon'] = df_valid_vessels['Lengdegrad'].astype(str).str.replace(',', '.').astype(float)

    return df_return