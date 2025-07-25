"""
Helper functions for fetching data from various sources such as Copernicus Marine and Norkyst800 (DEPRECATED).
"""

import os
import keyring
import numpy as np
import xarray as xr
import pandas as pd
import copernicusmarine
from logger import logger
# from utilities import latlon_to_XY

############################################
# Fetch data from Copernicus Marine
############################################
def check_credentials() -> bool:
    """
    Check if credentials for Copernicus Marine are available.
    If not, attempt to login using keyring.

    Returns:
        bool: True if credentials are available or the login is successful.
    """
    credentials_path = os.path.expanduser("~/.copernicusmarine/.copernicusmarine-credentials")
    if not os.path.exists(credentials_path):
        username = "pederbs1@gmail.com"
        password = keyring.get_password("CopernicusMarine", username)
        if password is None:
            raise ValueError("No password found in keyring for CopernicusMarine")
        return copernicusmarine.login(username, password)
    return True


def fetch_data_from_copernicus_marine(params: dict) -> xr.Dataset:
    """
    Extract data from Copernicus Marine using the Copernicus Marine API.
    The function extracts data from the specified dataset (specified by dataset_id) in parameters.

    Arguments:
        params (dict): Dictionary with the following keys:
            - dataset_id (str): The dataset identifier.
            - longitude (list): [min_longitude, max_longitude].
            - latitude (list): [min_latitude, max_latitude].
            - time (list): [start_datetime, end_datetime] in ISO format.
            - variables (list): List of variable names to request.
            - depths (list): [min_depth, max_depth] in meters.
    
    Returns:
        xr.Dataset: The resulting xarray dataset.
    """
    if not check_credentials():
        raise RuntimeError("Failed to authenticate with Copernicus Marine.")
    
    dataset = copernicusmarine.open_dataset(
        dataset_id=params["dataset_id"],
        minimum_longitude=params["longitude"][0],
        maximum_longitude=params["longitude"][1],
        minimum_latitude=params["latitude"][0],
        maximum_latitude=params["latitude"][1],
        start_datetime=params["time"][0],
        end_datetime=params["time"][1],
        variables=params["variables"],
        minimum_depth=params["depths"][0],
        maximum_depth=params["depths"][1]
    )
    return dataset


############################################
# Fetch data from Fiskeridirektoratet
############################################

# TODO: return xarray dataset??
def fetch_data_from_fiskeridirektoratet(params: dict, day_samples: bool = True, location_method: str = "both") -> pd.DataFrame:
    """
    Extract catch location and time from downloades ERS data from fiskeridirektoratet.
    The function reads the data from the csv files, filters the data based on the target species,
    calculates the location of the catch and returns the data as a pandas dataframe.
    (Implemented only for the species HER and MAC)

    Arguments:
        params (dict): Dictionary with the following keys:
            - dataset_path (str): The path to folder containing the datasets.
            - time (list): [start_datetime, end_datetime] in ISO format.
            - species (str): The fish species that is caught
            - location_method (str): The method used to determine the catch location ("start", "end", or "both")

    Returns:
        xr.Dataset: The resulting xarray dataset.
    """
    # List of columns containing data of interest
    info_cols = ["Starttidspunkt", "Stopptidspunkt", "Hovedart FAO (kode)",  
                 "Sildebestand (kode)", "Rundvekt", "Startposisjon bredde", 
                 "Startposisjon lengde", "Stopposisjon bredde", "Stopposisjon lengde",
                 "Fartøyidentifikasjon"
    ]

    # Extract the start and end years as integers
    start_year = int(params["time"][0].split("-")[0])
    end_year = int(params["time"][-1].split("-")[0])



    df = pd.DataFrame()
    logger.debug(f"Searching years: {start_year} - {end_year}")

    if params["species"].lower() == "her" or params["species"].lower() == "sild":
        search_col = "Sildebestand (kode)"
        target_species = "NOR01"
    elif params["species"].lower() == "mac" or params["species"].lower() == "makrell":
        search_col = "Hovedart FAO (kode)"
        target_species = "MAC"
    else:
        ValueError("Wrong species")
    logger.debug(f"Searching for fish species: {target_species}")
    

    # Iterate over every year from start_year to end_year (inclusive)
    for year in range(start_year, end_year + 1):
        # Construct the URL for the dataset
        data_url = params["dataset_path"] + "elektronisk-rapportering-ers-" + str(year) + \
                   "/elektronisk-rapportering-ers-" + str(year) + "-fangstmelding-dca.csv"
        data = pd.read_csv(data_url, delimiter=";", on_bad_lines="skip", skip_blank_lines=True, low_memory=False, na_values='na', parse_dates=True)

        # Remove columns that are not of interest
        data.drop(data.columns.difference(info_cols), axis=1, inplace=True)

        # Filter the data based on the target species
        data = data[data[search_col].str.contains(target_species, na=False)]

        # Remove entries outide of the time range specified in the parameters
        data = remove_outside_time_range(params, data)
        logger.debug(f"First and last entry for year {year}: {data['Starttidspunkt'].iloc[0]} - {data['Starttidspunkt'].iloc[-1]}")

        logger.debug(f"Data for year {year} shape: {data.shape}")
        # Append the data to the final dataset
        df = pd.concat([df, data])
    
    # Calculate the location of the catch
    logger.debug(f"Dataframe shape before calculating catch location: {df.shape}")
    df = calculate_catch_location(df, location_method)
    logger.debug(f"Dataframe shape after calculating catch location: {df.shape}")   

    # Remove rows with catch location outside of ROI
    df = df[~((df["CatchLat"] < params["SW"]["lat"]) | (df["CatchLat"] > params["NE"]["lat"]))]
    df = df[~((df["CatchLon"] < params["SW"]["lon"]) | (df["CatchLon"] > params["NE"]["lon"]))]

    logger.debug(f"Dataframe shape after removing outside ROI catch: {df.shape}")

    # Remove the unnecessary columns
    df.drop(
        columns=[
            # "Startposisjon bredde", "Startposisjon lengde", "Stopposisjon bredde", "Stopposisjon lengde", 
            "Hovedart FAO (kode)", "Sildebestand (kode)"
        ], 
        inplace=True
    )

    if day_samples: # TODO: Find best way to select date (e.g. random, first, last)
        # Group the data by date
        logger.debug(f"Dataframe shape: {df.shape}")
        df = df.groupby("Date", as_index=False).agg(lambda x: list(x))
        logger.debug(f"Dataframe shape after grouping: {df.shape}")
    else:
        # TODO: Implement some functionality (not sure what)
        pass

    
    # Remove the rest of the unnecessary columns
    # df.drop(
    #     columns=["Starttidspunkt", "Stopptidspunkt"], 
    #     inplace=True
    # )

    logger.info(f"Final dataset shape: {df.shape}")

    # Convert the DataFrame to an xarray dataset
    # ds = df.to_xarray()

    return df


def calculate_catch_location(df: pd.DataFrame, location_method: str = "start") -> pd.DataFrame:
    """
    Helper function for extracting the locaton of the catch from ERS data from fiskeridirektoratet.
    As Fiskeridirektoratet provides both start and end locations for the fishing activity, three methods for 
    choosing the location of the catch are implemented.

    Parameters:
      data_frame (pd.DataFrame): Pandas Data Frame.
      location_method (str): Method for choosing the location ('start', 'avg', slutt). (avg uses start time for Date)
      WARNING: avg gives unnatural values (bug catches on land)

    Returns:
      pd.DataFrame
    """
    # Replace commas with dots in the columns with GPS coordinates
    cols = ["Startposisjon bredde", "Startposisjon lengde", "Stopposisjon bredde", "Stopposisjon lengde"]
    df[cols] = df[cols].replace({',': '.'}, regex=True)
    

    # Convert the coordinate columns to float
    df[cols] = df[cols].astype(float)

    if location_method == "start":
        df["CatchLat"] = df["Startposisjon bredde"]
        df["CatchLon"] = df["Startposisjon lengde"]
        # OLD
        df["Location"] = list(zip(df["Startposisjon bredde"], df["Startposisjon lengde"]))
        # Rewrite the date to a more readable format 
        df["Date"] = df["Starttidspunkt"].str[6:10] + \
                    '.' + df["Starttidspunkt"].str[3:5] + \
                    '.' + df["Starttidspunkt"].str[:2]
        

    elif location_method == "slutt":
        df["CatchLat"] = df["Stopposisjon bredde"]
        df["CatchLon"] = df["Stopposisjon lengde"]
        # OLD
        df["Location"] = list(zip(df["Stopposisjon bredde"], df["Stopposisjon lengde"]))
        # Rewrite the date to a more readable format 
        df["Date"] = df["Stopptidspunkt"].str[6:10] + \
                    '.' + df["Stopptidspunkt"].str[3:5] + \
                    '.' + df["Stopptidspunkt"].str[:2]

    elif location_method == "avg":
        logger.warning(f"\nWARNING: using avg for catch location my give unnatural values\n")
        df["CatchLat"] = (df["Startposisjon bredde"] + df["Stopposisjon bredde"]) / 2
        df["CatchLon"] = (df["Startposisjon lengde"] + df["Stopposisjon lengde"]) / 2
        # OLD
        avg_bredde = (df["Startposisjon bredde"] + df["Stopposisjon bredde"]) / 2
        avg_lengde = (df["Startposisjon lengde"] + df["Stopposisjon lengde"]) / 2
        df["Location"] = list(zip(avg_bredde, avg_lengde))
        # Rewrite the date to a more readable format 
        df["Date"] = df["Starttidspunkt"].str[6:10] + \
                    '.' + df["Starttidspunkt"].str[3:5] + \
                    '.' + df["Starttidspunkt"].str[:2]

    elif location_method == "both":
        df_start = df.copy()
        df_start["CatchLat"] = df_start["Startposisjon bredde"]
        df_start["CatchLon"] = df_start["Startposisjon lengde"]
        df_start["Method"] = "start"
        # OLD
        df_start["Location"] = list(zip(df_start["Startposisjon bredde"], df_start["Startposisjon lengde"]))
        # Rewrite the date to a more readable format 
        df_start["Date"] = df_start["Starttidspunkt"].str[6:10] + \
                    '.' + df_start["Starttidspunkt"].str[3:5] + \
                    '.' + df_start["Starttidspunkt"].str[:2]

        df_end = df.copy()
        df_end["CatchLat"] = df_end["Stopposisjon bredde"]
        df_end["CatchLon"] = df_end["Stopposisjon lengde"]
        df_end["Method"] = "stop"
        # OLD
        df_end["Location"] = list(zip(df_end["Stopposisjon bredde"], df_end["Stopposisjon lengde"]))
        # Rewrite the date to a more readable format 
        df_end["Date"] = df_end["Stopptidspunkt"].str[6:10] + \
                    '.' + df_end["Stopptidspunkt"].str[3:5] + \
                    '.' + df_end["Stopptidspunkt"].str[:2]

        # Concatenate the two dataframes
        df = pd.concat([df_start, df_end], ignore_index=True)
    else:
        ValueError(f"Unsuported input {location_method}")
    
    return df


def remove_outside_time_range(params: dict, data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove dates outside of the specified time range.
    """

    # Extract target month/date range
    start_date = pd.to_datetime(params["time"][0], format='mixed', dayfirst=True)
    end_date = pd.to_datetime(params["time"][1], format='mixed', dayfirst=True)

    # Parse once up front
    data['date'] = pd.to_datetime(
        data['Starttidspunkt'],
        dayfirst=True,
        format='mixed',    # auto‑detect date vs datetime
        errors='coerce'    # convert unparsable to NaT
    )

    # Count the failures
    n_bad = data['date'].isna().sum()
    if n_bad > 0:
        logger.warning(f"{n_bad} / {len(data)} rows failed to parse Starttidspunkt → dropped")

    # Extract month/day ints
    month = data['date'].dt.month
    day   = data['date'].dt.day
    start = start_date
    end   = end_date

    data = data.drop(columns=['date'])

    # Build mask: (month,day) ≥ (start.month,start.day) AND ≤ (end.month,end.day)
    mask = (
        ((month > start.month) | ((month == start.month) & (day >= start.day))) &
        ((month < end.month)   | ((month == end.month)   & (day <= end.day)))
    )

    return data.loc[mask]


############################################
# Fetch data from norkyst800
############################################
def get_norkyst800_data(params: dict) -> xr.Dataset:
    """
    Get dataset for the specified variables and unique time points.
    If a requested date is not available in the dataset's time coordinate,
    the previous available date will be used instead.
    
    Parameters:
        params (dict): Dictionary with keys:
            - "variables": a list of variable names to extract.
            - "time": a list of unique date strings (e.g., ["2023-01-01", "2023-01-05", ...]).
                      For any date that is not available, the previous available date is used.
            - "depth": Number for the lowest depth layer to keep
    
    Returns:
        xr.Dataset: The subset of the dataset for the provided variables and adjusted time points.
    """
    url = "https://thredds.met.no/thredds/dodsC/sea/norkyst800m/24h/aggregate_be"
    ds = xr.open_dataset(url)


    # TODO: make the bounds passable as arguments
    lon = ds.lon
    lat = ds.lat

    # get index bounds
    y0, y1, x0, x1 = bbox_indices(lon, lat, params)
    logger.debug(f"X range: {x0} -> {x1}")
    logger.debug(f"Y range: {y0} -> {y1}")
    
    # Convert requested dates (strings) to np.datetime64 objects
    requested_dates = [np.datetime64(d, 'D') + np.timedelta64(12, 'h') for d in params["time"]]
    
    # Get available dates from the dataset and ensure they're sorted
    available_dates = np.sort(ds.time.values)
    not_exist_n = 0

    selected_dates = []
    for r in requested_dates:
        if r in available_dates:
            selected_dates.append(r)
        else:
            logger.error(f"date: {r} does not exist...")
            not_exist_n += 1
            # Find all available dates that are before the requested date
            earlier = available_dates[available_dates < r]
            if earlier.size > 0:
                # Use the latest available date that is less than the requested date
                selected_dates.append(earlier.max())
                logger.info(f"using: {earlier.max()} instead.")
            else:
                # If no earlier date exists, use the earliest available date
                selected_dates.append(available_dates[0])
                logger.info(f"using: {available_dates[0]} instead.")

    logger.debug(f"Replaced {not_exist_n} dates")
    if not_exist_n > 0:
        logger.debug(f"Replaced date: {requested_dates} with {selected_dates}")
    
    data = ds[params["variables"]].sel(
        time=selected_dates, 
        depth=params["depth"],
    )
    data = data.isel(
        X=slice(x0, x1+1),
        Y=slice(y0, y1+1)
    )
    return data


def find_nearest_index(lon, lat, lon_pt, lat_pt):
    """
    Return the (y,x) index whose (lon,lat) is closest to (lon_pt,lat_pt).
    """
    d2  = (lon - lon_pt)**2 + (lat - lat_pt)**2
    idx = int(d2.values.argmin())
    y, x = np.unravel_index(idx, lon.shape)
    # logger.debug(f"Nearest (Y,X) to ({lon_pt},{lat_pt}) → ({y},{x})")
    return y, x

def bbox_indices(lon, lat, corners):
    """
    Given 2D lon/lat arrays and a corners dict,
    returns (y0, y1, x0, x1) so that every corner is inside that index box.

    corners should be a dict with keys "SW","SE","NE","NW", each
    mapping to {"lon":..., "lat":...}.
    """
    # find each corner's (y,x)
    idxs = []
    for name in ("SW","SE","NE","NW"):
        p = corners[name]
        yi, xi = find_nearest_index(lon, lat, p["lon"], p["lat"])
        idxs.append((yi, xi))

    ys, xs = zip(*idxs)
    y0, y1 = min(ys), max(ys)
    x0, x1 = min(xs), max(xs)

    logger.info(f"Bounding indices Y: {y0}→{y1}, X: {x0}→{x1}")
    return y0, y1, x0, x1

if __name__ == "__main__":
    pass
    