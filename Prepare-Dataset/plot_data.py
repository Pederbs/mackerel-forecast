"""
Helper functions for plotting data and making images for datasets.
"""

import os
import shutil
import numpy as np
import xarray as xr
import pandas as pd
from logger import logger
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature

def plot_copernicus_variable(ds, variable="temperature", out_folder=None, region_coords=None, depth: int = 0) -> None:
    """
    DEPRICATED
    Plot one or more variables from an xarray dataset.

    Parameters:
        dataset (xarray.Dataset or str): An already opened xarray.Dataset or a path to a netCDF file.
        variables (str or list of str, optional): Name(s) of the variable(s) to plot.
            If not provided, the first variable in the dataset is plotted.
        save (bool, optional): If True, saves each plot as a PNG file.
        save_path (str, optional): Directory or filename prefix to use when saving plots.

    Returns:
        None
    """
    # if os.path.exists(out_folder):
    #     shutil.rmtree(out_folder)
    # os.makedirs(out_folder, exist_ok=True)

    # If variable is a list, iterate over each one.
    variables = variable if isinstance(variable, list) else [variable]

    for var_name in variables:
        # Extract the DataArray for the current variable
        var = ds[var_name]

        # Try standard names for lat/lon
        lat = ds.get("latitude")
        lon = ds.get("longitude")

        # If time exists, loop over time steps
        times = ds["time"].values if "time" in var.dims else [None]

        for i, t in enumerate(times):
            if t is not None:
                var_t = var.sel(time=t)
                t_dt = pd.to_datetime(t)
                t_str = t_dt.strftime("%Y-%m-%d")
            else:
                var_t = var
                t_str = f"frame_{i}"

            # Select first depth if present
            if "depth" in var_t.dims:
                var_t = var_t.sel(depth=depth)

            # Select any other extra dims
            for dim in var_t.dims:
                if dim not in {"lat", "latitude", "lon", "longitude", "Y", "X"}:
                    var_t = var_t.isel({dim: 0})

            # Meshgrid if needed
            if lat.ndim == 1 and lon.ndim == 1:
                lon2d, lat2d = np.meshgrid(lon, lat)
            else:
                lon2d, lat2d = lon, lat

            fig = plt.figure(figsize=(2, 1))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

            # Apply region cropping
            if region_coords is not None:
                extent = [
                    region_coords["SW"]["lon"],  # min lon
                    region_coords["NE"]["lon"],  # max lon
                    region_coords["SW"]["lat"],  # min lat
                    region_coords["NE"]["lat"],  # max lat
                ]
                ax.set_extent(extent, crs=ccrs.PlateCarree())

            try:
                mesh = ax.pcolormesh(
                    lon2d,
                    lat2d,
                    var_t.values,  # make sure we pass a NumPy array
                    transform=ccrs.PlateCarree(),
                    cmap="binary",
                    # TODO: set vmin/vmax based on variable
                    vmin=0,
                    vmax=1
                )
            except Exception as e:
                logger.warning(f"Failed to plot {var_name} for time {t_str}. Error: {e}")
                continue

            # No colorbar, no title, no axes
            ax.axis("off")
            if out_folder is not None:
                # Save the figure
                out_path = os.path.join(out_folder, f"{var_name}_{t_str}.png")
                save_figure(out_path, ax)
                logger.info(f"Saved: {out_path}")
            else:
                for spine in ax.spines.values():
                    spine.set_visible(False)
                plt.show()
            plt.close()

def extract_copernicus_variable(ds, variable="temperature", depth: float = 0, region_coords: dict = None, keep_nan: bool = False) -> dict:
    """
    Extract a 2D NumPy array (or arrays, if multiple variables) of the given variable at the specified depth.
    If region_coords is given, values outside the bounding box will be set to NaN.

    Returns:
        dict: mapping variable name -> 2D NumPy array of shape (lat, lon)
    """
    if region_coords is not None:
        ds = ds.sel(latitude=slice(region_coords["SW"]["lat"], region_coords["NE"]["lat"]),
                    longitude=slice(region_coords["SW"]["lon"], region_coords["NE"]["lon"]))
    # ensure we always have a list of variable names
    variables = variable if isinstance(variable, list) else [variable]
    result = {}

    # get amount of datapoints in lat and lon dimensions
    lat_dim = ds.dims["latitude"]
    lon_dim = ds.dims["longitude"]

    #create an empty array for each variable
    for var in variables:
        result[var] = np.full((lat_dim, lon_dim), np.nan, dtype=np.float32)
        # populate the array with the data at the specified depth
        if var in ds:
            result[var] = ds[var].sel(depth=depth, method='nearest')
            result[var] = result[var]
            # flip the array to match the expected orientation
            result[var] = np.flipud(result[var])
            if not keep_nan:
                # replace NaN values with 0
                result[var] = np.nan_to_num(result[var], nan=0.0)

            # logger.info(f"type: {type(result[var])}, shape: {result[var].shape}")
    return result

def plot_all_catches(df, region_coords, save_path=None, use_catch_weight=False) -> None:
    """
    Plot all fishing catches from the provided dataset.

    Each row in data (which can be a pandas DataFrame or xarray Dataset) is expected to have:
      - a "location" column: an array (list) of tuples (lat, lon), or a single tuple,
      - a "Rundvekt" column: a list of catch weights (floats) where each weight corresponds to the catch location.
    
    If use_catch_weight is True, the marker size is scaled by the catch weight (divided by 1000 as an example scale).

    Parameters:
        data (pandas.DataFrame or xarray.Dataset): Data containing fishing records.
        region_coords (dict): Dictionary plot borders (see area.json)
        save_path (str): File path or filename prefix to use when saving the plot.
        use_catch_weight (bool): If True, scales marker size by catch weight.

    Returns:
        None
    """
    #TODO: check that the image is not cropped weirdly
    lats = []
    lons = []
    sizes = []
    
    for idx, row in df.iterrows():
        # Process the location data.
        # loc_data = row["Location"]
        loc_data = (row["CatchLat"], row["CatchLon"])
        # Determine if we have a single location or a list of them.
        if isinstance(loc_data, list):
            catch_locations = loc_data
        elif isinstance(loc_data, tuple):
            # Check if this tuple is a single catch (both elements convertible to float)
            if len(loc_data) == 2:
                try:
                    float(loc_data[0])
                    float(loc_data[1])
                    catch_locations = [loc_data]
                except Exception:
                    # Otherwise, if not, try converting it to a list
                    catch_locations = list(loc_data)
            else:
                catch_locations = list(loc_data)
        else:
            # Wrap non-list, non-tuple entries in a list.
            catch_locations = [loc_data]
        
        # Process the catch weight(s).
        weight_data = row["Rundvekt"]
        if isinstance(weight_data, list):
            catch_weights = weight_data
        elif isinstance(weight_data, tuple):
            catch_weights = list(weight_data)
        else:
            catch_weights = [weight_data]
        
        # Loop over each catch in the row.
        for i, loc in enumerate(catch_locations):
            try:
                # Expect loc to be a tuple (lat, lon)
                if isinstance(loc, (list, tuple)) and len(loc) == 2:
                    lat = float(loc[0])
                    lon = float(loc[1])
                else:
                    # Fallback: if loc is a string with comma-separated values
                    lat_str, lon_str = loc.split(',')
                    lat = float(lat_str)
                    lon = float(lon_str)
            except Exception as e:
                logger.debug(f"Skipping row {idx} element {i} due to error in location: {e}")
                continue
            try:
                # Use the corresponding weight if available.
                weight = float(catch_weights[i]) if i < len(catch_weights) else 0.0
            except Exception as e:
                logger.debug(f"Skipping row {idx} element {i} due to error in Rundvekt: {e}")
                continue
            
            lats.append(lat)
            lons.append(lon)
            sizes.append(weight / 10000 if use_catch_weight else 10)
    
    # Set up the map with a PlateCarree projection.
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots( subplot_kw={'projection': proj})
    
    # Define the map extent using region boundaries.
    extent = [
        region_coords["SW"]["lon"],  # minimum longitude
        region_coords["NE"]["lon"],  # maximum longitude
        region_coords["SW"]["lat"],  # minimum latitude
        region_coords["NE"]["lat"]   # maximum latitude
    ]
    ax.set_extent(extent, crs=proj)
    
    # Add map features.
    ax.add_feature(cfeature.LAND, facecolor='black')
    # ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    
    # Scatter plot all fishing catch positions.
    ax.scatter(lons, lats, transform=proj, s=sizes, color='red', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot if requested.
    if save_path is not None:
        save_figure(save_path, ax, dpi=500)
        logger.info(f"Plot saved as {save_path}")
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.show()
    plt.close()

def plot_daily_catches(df, region_coords, save_dir=None, use_catch_weight=True) -> None:
    """
    Plot the fishing catches for each day from the provided dataset.
    
    The DataFrame is expected to have:
      - "Date": a column with date strings (e.g., "2023.01.02"),
      - "location": a column with an array (list) of tuples (lat, lon) or a single tuple,
      - "Rundvekt": a column with a list of catch weights (floats) corresponding to each catch location.
    
    For each unique day, a separate map is created showing the catch positions. If use_catch_weight is True,
    marker size is scaled by the catch weight.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing fishing records.
        region_coords (dict): Dictionary defining the region boundaries, e.g.,
                              {"SW": {"lat": min_lat, "lon": min_lon}, "NE": {"lat": max_lat, "lon": max_lon}}.
        save_dir (str): Directory where each day's plot will be saved (optional).
        use_catch_weight (bool): If True, scales marker size by catch weight.
    
    Returns:
        None
    """
    # # TODO: NOT saving to same size as the other pictures....
    # if os.path.exists(save_dir):
    #     shutil.rmtree(save_dir)
    # # Ensure save_dir exists if provided.
    # if save_dir is not None:
    #     os.makedirs(save_dir, exist_ok=True)

    # Get the unique dates from the DataFrame.
    unique_dates = sorted(df["Date"].unique())

    for date in unique_dates:
        # Filter the DataFrame for the given date.
        daily_df = df[df["Date"] == date]
        
        # Drop rows with missing latitude or longitude. (avoid user waring)
        daily_df = daily_df.dropna(subset=["CatchLat", "CatchLon"])
        
        # Convert the relevant columns to lists.
        lats = daily_df["CatchLat"].astype(float).tolist()
        lons = daily_df["CatchLon"].astype(float).tolist()
        # Assuming 'Rundvekt' is the catch weight.
        weights = daily_df["Rundvekt"].astype(float).tolist() if "Rundvekt" in daily_df.columns else [0]*len(lats)
        
        # Set up the map with a PlateCarree projection.
        proj = ccrs.PlateCarree()
        fig_const = 1
        fig, ax = plt.subplots(figsize=(fig_const, fig_const), subplot_kw={'projection': proj})

        extent = [
            region_coords["SW"]["lon"],  # minimum longitude
            region_coords["NE"]["lon"],  # maximum longitude
            region_coords["SW"]["lat"],  # minimum latitude
            region_coords["NE"]["lat"]   # maximum latitude
        ]
        ax.set_extent(extent, crs=proj)
        
        # Add map features.

        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='black')
        # ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='grey')

        # Scatter plot the catch positions using catch weight as the color.
        if lats and lons:  # Only plot if there are catches.
            logger.info(f"Plotting catches for date: {date}")

            if use_catch_weight:
                # Normalize weights for better visualization.
                norm_weights = np.array(weights) / np.max(weights) * 1
                ax.scatter(lons, lats, transform=proj,s=1, c=norm_weights, cmap='viridis', alpha=1.0)
                # fig.colorbar(scatter, ax=ax, label="Catch Weight")
            else:
                ax.scatter(lons, lats, s=1, c='white', transform=proj, alpha=1.0)
            # Add a colorbar to show the catch weight scale.
            # fig.colorbar(scatter, ax=ax, label="Catch Weight")
        
        plt.tight_layout()
        
        # Save or display the plot.
        if save_dir is not None:
            # Create a safe filename for the date.
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            filename = os.path.join(save_dir, f"catch_{date_str}.png")
            save_figure(filename, ax)
            # logger.info(f"Plot saved as {filename}")
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)
            plt.show()
        plt.close(fig)




def plot_norkyst800_temperature(ds, out_folder = None, region_coords=None, depth=0) -> None:
    """
    DEPRICATED
    Plot surface temperature from a Norkyst800 dataset for each time step,
    and optionally crop to a specified region using region_coords.
    
    Parameters:
        ds (xr.Dataset):
            - 'temperature' (time, depth, Y, X)
            - 'lon' (Y, X)
            - 'lat' (Y, X)
            - 'time' (time)
            - 'depth' (depth)
        out_folder (str): Path to the folder where plots will be saved.
        region_coords (dict, optional): Dictionary defining the bounding box:
            {
              "SW": {"lat": 66.21, "lon": 7.33},
              "SE": {"lat": 66.21, "lon": 17.90},
              "NE": {"lat": 69.00, "lon": 17.90},
              "NW": {"lat": 69.00, "lon": 7.33}
            }
            If provided, the plot will be cropped to this region.
    """
    # if os.path.exists(out_folder):
    #     shutil.rmtree(out_folder)
    # # Ensure save_dir exists if provided.
    # if out_folder is not None:
    #     os.makedirs(out_folder, exist_ok=True)

    # Extract the 2D lat/lon arrays
    lat = ds['lat']
    lon = ds['lon']

    # Loop over each time step, adding a counter for unique filenames
    for i, t in enumerate(ds['time'].values):
        t_dt = pd.to_datetime(t)  # convert to pandas datetime

        # Select the surface layer (depth=0)
        use_depth = depth
        temp = ds['temperature'].sel(time=t).sel(depth=use_depth)

        # Create a figure with a PlateCarree projection
        fig = plt.figure(figsize=(2, 1))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # If region_coords is provided, set the map extent to crop the view
        if region_coords is not None:
            extent = [
                region_coords["SW"]["lon"],  # min lon
                region_coords["NE"]["lon"],  # max lon
                region_coords["SW"]["lat"],  # min lat
                region_coords["NE"]["lat"]   # max lat
            ]
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Add map features
        # ax.add_feature(cfeature.LAND, facecolor='black')
        # ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

        # Plot the temperature
        try:
            mesh = ax.pcolormesh(
                lon, lat, temp,
                transform=ccrs.PlateCarree(),
                cmap='binary',
                # shading='auto'
                vmin=0,
                vmax=1
            )
        except Exception as e:
            logger.warning(f"Failed to plot temperature for time {t_dt}. Error: {e}")
            continue

        
        if out_folder is not None:
            # Generate a unique filename using the counter and time
            out_filename = os.path.join(out_folder, f"temp_h{use_depth}_{t_dt.strftime('%Y-%m-%d')}.png")
            save_figure(out_filename, ax)
            logger.info(f"Plot saved as {out_filename}")
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)
            plt.show()
        plt.close(fig)

def plot_norkyst800_salinity(ds, out_folder = None, region_coords=None, depth=0) -> None:
    """
    DEPRICATED
    Plot surface salinity from a Norkyst800 dataset for each time step,
    and optionally crop to a specified region using region_coords.
    
    Parameters:
        ds (xr.Dataset):
            - 'salinity' (time, depth, Y, X)
            - 'lon' (Y, X)
            - 'lat' (Y, X)
            - 'time' (time)
            - 'depth' (depth)
        out_folder (str): Path to the folder where plots will be saved.
        region_coords (dict, optional): Dictionary defining the bounding box:
            {
              "SW": {"lat": 66.21, "lon": 7.33},
              "SE": {"lat": 66.21, "lon": 17.90},
              "NE": {"lat": 69.00, "lon": 17.90},
              "NW": {"lat": 69.00, "lon": 7.33}
            }
            If provided, the plot will be cropped to this region.
    """
    # if os.path.exists(out_folder):
    #     shutil.rmtree(out_folder)
    # # Ensure save_dir exists if provided.
    # if out_folder is not None:
    #     os.makedirs(out_folder, exist_ok=True)

    # Extract the 2D lat/lon arrays
    lat = ds['lat']
    lon = ds['lon']

    # Loop over each time step, adding a counter for unique filenames
    for i, t in enumerate(ds['time'].values):
        t_dt = pd.to_datetime(t)  # convert to pandas datetime

        # Select the surface layer (depth=0)
        use_depth = depth
        salinity = ds['salinity'].sel(time=t).sel(depth=use_depth)

        # Create a figure with a PlateCarree projection
        fig = plt.figure(figsize=(2, 1))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # If region_coords is provided, set the map extent to crop the view
        if region_coords is not None:
            extent = [
                region_coords["SW"]["lon"],  # min lon
                region_coords["NE"]["lon"],  # max lon
                region_coords["SW"]["lat"],  # min lat
                region_coords["NE"]["lat"]   # max lat
            ]
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Add map features
        # ax.add_feature(cfeature.LAND, facecolor='black')
        # ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

        # Plot the salinity
        try:
            mesh = ax.pcolormesh(
                lon, lat, salinity,
                transform=ccrs.PlateCarree(),
                cmap='binary', # higher values are darker
                # shading='auto'
                # TODO: check the salinity range (current range is guesswork)
                vmin=0,
                vmax=1
            )
        except Exception as e:
            logger.warning(f"Failed to plot salinity for time {t_dt}. Error: {e}")
            continue


        if out_folder is not None:
            # Generate a unique filename using the counter and time
            out_filename = os.path.join(out_folder, f"sal_h{use_depth}_{t_dt.strftime('%Y-%m-%d')}.png")
            save_figure(out_filename, ax)
            logger.info(f"Plot saved as {out_filename}")
        else:
            for spine in ax.spines.values():
                spine.set_visible(False)
            plt.show()
        plt.close(fig)

def save_figure(file: str, ax: plt.Axes, dpi: int = 512) -> None:
    """
    Helper function for saving the figure with a standard format
    """

    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.savefig(file, dpi=dpi, bbox_inches='tight', pad_inches=0)



# Example usage:
if __name__ == '__main__':
    pass
    