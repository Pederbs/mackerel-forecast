"""
Script for finding valid ocean locations and extracting environmental data.
Used for the correlation analysis presented in the thesis.
"""

import os
import random
import datetime
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.spatial.distance import cdist
from tqdm import tqdm
from utilities import get_region_border


def find_days_interval(start: str = '06-01-2011', end: str = '11-30-2024'):
    """Finds the date range between two dates for every year"""
    all_dates_set = set()
    start_year = start[-4:]
    end_year = end[-4:]
    for year in range(int(start_year), int(end_year) + 1):
        start_date = datetime.date(year, int(start[:2]), int(start[3:5]))
        end_date = datetime.date(year, int(end[:2]), int(end[3:5]))
        dates_in_year = pd.date_range(start=start_date, end=end_date)
        all_dates_set.update(dates_in_year)
    return all_dates_set


def load_environmental_data(date_str, bio_path, phy_path):
    """Load biological and physical environmental data for a specific date"""
    bio_file = os.path.join(bio_path, f"{date_str}.nc")
    phy_file = os.path.join(phy_path, f"{date_str}.nc")
    
    bio_data = None
    phy_data = None
    
    if os.path.exists(bio_file):
        bio_data = xr.open_dataset(bio_file)
    if os.path.exists(phy_file):
        phy_data = xr.open_dataset(phy_file)
    
    return bio_data, phy_data


def is_valid_location_improved(lat, lon, bio_data, phy_data):
    """Check if a location is valid (not on land, has environmental data)"""
    
    # Check bio data
    if bio_data is not None:
        try:
            bio_point = bio_data.sel(latitude=lat, longitude=lon, method='nearest')
            # Check if any variable at any depth has valid data
            has_valid_bio = False
            for var in bio_data.data_vars:
                var_values = bio_point[var].values
                if not np.isnan(var_values).all():
                    has_valid_bio = True
                    break
            if not has_valid_bio:
                return False
        except Exception as e:
            print(f"Error checking bio data at {lat}, {lon}: {e}")
            return False
    
    # Check phy data
    if phy_data is not None:
        try:
            phy_point = phy_data.sel(latitude=lat, longitude=lon, method='nearest')
            # Check if any variable at any depth has valid data
            has_valid_phy = False
            for var in phy_data.data_vars:
                var_values = phy_point[var].values
                if not np.isnan(var_values).all():
                    has_valid_phy = True
                    break
            if not has_valid_phy:
                return False
        except Exception as e:
            print(f"Error checking phy data at {lat}, {lon}: {e}")
            return False
    
    return True


def generate_valid_location_pool(region, pool_size=1000, sample_date=None, bio_path=None, phy_path=None):
    """
    Generate a pool of valid random locations within the region.
    
    This function creates a pre-computed pool of valid ocean locations that can be used
    for selecting no-catch points. It validates each location against environmental data
    to ensure they are not on land and have available data.
    
    Note: This happens ONCE at the start, not for each day!
    """
    print(f"Generating pool of {pool_size} valid locations...")
    
    # If sample_date is provided, use it to validate locations
    bio_data, phy_data = None, None
    if sample_date and bio_path and phy_path:
        print(f"Loading environmental data for validation using date: {sample_date}")
        bio_data, phy_data = load_environmental_data(sample_date, bio_path, phy_path)
        
        if bio_data is None and phy_data is None:
            print(f"Warning: No environmental data found for {sample_date}")
            print("Generating locations without validation...")
        else:
            print(f"Bio data available: {bio_data is not None}")
            print(f"Phy data available: {phy_data is not None}")
    
    valid_locations = []
    attempts = 0
    max_attempts = pool_size * 20  # Allow more attempts
    
    while len(valid_locations) < pool_size and attempts < max_attempts:
        # Generate random point within region
        lat = random.uniform(region["SW"]["lat"], region["NE"]["lat"])
        lon = random.uniform(region["SW"]["lon"], region["NE"]["lon"])
        
        # If we have environmental data, check if location is valid
        if bio_data is not None or phy_data is not None:
            if is_valid_location_improved(lat, lon, bio_data, phy_data):
                valid_locations.append([lat, lon])
        else:
            # If no environmental data available, just add the location
            valid_locations.append([lat, lon])
        
        attempts += 1
        
        # Print progress every 2000 attempts
        if attempts % 2000 == 0:
            print(f"Progress: {attempts} attempts, {len(valid_locations)} valid locations found so far...")
    
    # Close datasets to free memory
    if bio_data is not None:
        bio_data.close()
    if phy_data is not None:
        phy_data.close()
    
    print(f"Successfully generated {len(valid_locations)} valid locations out of {pool_size} requested (took {attempts} attempts)")
    return valid_locations


def get_no_catch_points_from_pool(catch_locations, location_pool, num_points, min_distance=0.1):
    """
    Select no-catch points from the pre-generated pool.
    
    This function SELECTS (not generates) points from the existing pool,
    ensuring minimum distance from catch locations and other selected points.
    """
    # Shuffle the pool to get random selection
    shuffled_pool = location_pool.copy()
    random.shuffle(shuffled_pool)
    
    selected_points = []
    
    for lat, lon in shuffled_pool:
        if len(selected_points) >= num_points:
            break
            
        # Check minimum distance from catch locations
        if len(catch_locations) > 0:
            distances = cdist([[lat, lon]], catch_locations)
            if np.min(distances) < min_distance:
                continue
        
        # Check minimum distance from already selected no-catch points
        if len(selected_points) > 0:
            distances = cdist([[lat, lon]], selected_points)
            if np.min(distances) < min_distance:
                continue
        
        selected_points.append([lat, lon])
    
    return selected_points


def extract_environmental_variables(lat, lon, bio_data, phy_data):
    """Extract all environmental variables at a given location"""
    variables = {}
    
    if bio_data is not None:
        try:
            bio_point = bio_data.sel(latitude=lat, longitude=lon, method='nearest')
            for var in bio_data.data_vars:
                for depth in bio_data.depth.values:
                    var_name = f"bio_{var}_depth_{depth}"
                    try:
                        value = bio_point[var].sel(depth=depth).values
                        # Extract scalar value properly to avoid deprecation warning
                        if np.isscalar(value):
                            scalar_value = float(value)
                        else:
                            scalar_value = float(value.item())
                        variables[var_name] = scalar_value if not np.isnan(scalar_value) else None
                    except:
                        variables[var_name] = None
        except Exception as e:
            print(f"Error extracting bio data at {lat}, {lon}: {e}")
    
    if phy_data is not None:
        try:
            phy_point = phy_data.sel(latitude=lat, longitude=lon, method='nearest')
            for var in phy_data.data_vars:
                for depth in phy_data.depth.values:
                    var_name = f"phy_{var}_depth_{depth}"
                    try:
                        value_array = phy_point[var].sel(depth=depth).values
                        # Handle both scalar and array cases
                        if np.isscalar(value_array):
                            value = float(value_array)
                        else:
                            value = float(value_array.item())
                        variables[var_name] = value if not np.isnan(value) else None
                    except:
                        variables[var_name] = None
        except Exception as e:
            print(f"Error extracting phy data at {lat}, {lon}: {e}")
    
    return variables


def method1_implementation(df_catch, region, bio_path, phy_path, min_distance=0.1, pool_size=1000, save_pool=True, pool_output_path=None):
    """
    Method 1: For every day, extract environmental variables for catch locations
    and generate corresponding no-catch points with same environmental data
    
    Args:
        df_catch: DataFrame with catch data
        region: Dictionary with region boundaries
        bio_path: Path to biological data files
        phy_path: Path to physical data files
        min_distance: Minimum distance between catch and no-catch points
        pool_size: Size of the pre-generated location pool
        save_pool: Whether to save the location pool to CSV
        pool_output_path: Path to save the location pool CSV file
    
    Returns:
        DataFrame with balanced catch and no-catch records
    """
    all_data = []
    
    # Generate a pool of valid locations using a sample date
    sample_date = df_catch['Date'].iloc[0].strftime('%Y-%m-%d')
    valid_location_pool = generate_valid_location_pool(
        region, pool_size=pool_size, sample_date=sample_date, 
        bio_path=bio_path, phy_path=phy_path
    )
    
    if len(valid_location_pool) == 0:
        print("Warning: No valid locations found in the pool!")
        return pd.DataFrame()
    
    print(f"Using location pool with {len(valid_location_pool)} valid locations")
    
    # Save the location pool to CSV if requested
    if save_pool and pool_output_path:
        pool_df = pd.DataFrame(valid_location_pool, columns=['Latitude', 'Longitude'])
        pool_df['LocationType'] = 'ValidPool'
        pool_df['Region'] = region.get('name', 'unknown')
        pool_df['PoolSize'] = pool_size
        pool_df['GeneratedDate'] = pd.Timestamp.now()
        pool_df.to_csv(pool_output_path, index=False, sep=';')
        print(f"Location pool saved to: {pool_output_path}")
    elif save_pool:
        print("Warning: save_pool=True but no pool_output_path provided")
    
    # Group by date
    for date, day_data in tqdm(df_catch.groupby('Date'), desc="Processing days"):
        date_str = date.strftime('%Y-%m-%d')
        
        # Load environmental data for this date
        bio_data, phy_data = load_environmental_data(date_str, bio_path, phy_path)
        
        if bio_data is None and phy_data is None:
            print(f"No environmental data found for {date_str}")
            continue
        
        # Extract catch locations
        catch_locations = day_data[['CatchLat', 'CatchLon']].values.tolist()
        num_catches = len(catch_locations)
        
        # Process catch locations
        for _, catch_row in day_data.iterrows():
            lat, lon = catch_row['CatchLat'], catch_row['CatchLon']
            
            # Extract environmental variables
            env_vars = extract_environmental_variables(lat, lon, bio_data, phy_data)
            
            # Create record
            record = {
                'Date': date,
                'Latitude': lat,
                'Longitude': lon,
                'HasCatch': 1,
                'CatchWeight': catch_row.get('Rundvekt', 0),
                **env_vars
            }
            all_data.append(record)
        
        # Get no-catch points from the pool
        no_catch_points = get_no_catch_points_from_pool(
            catch_locations, valid_location_pool, num_catches, min_distance
        )
        
        # print(f"Date {date_str}: {len(catch_locations)} catches, {len(no_catch_points)} no-catch points selected from pool")
        
        # Process no-catch locations
        for lat, lon in no_catch_points:
            # Extract environmental variables
            env_vars = extract_environmental_variables(lat, lon, bio_data, phy_data)
            
            # Create record
            record = {
                'Date': date,
                'Latitude': lat,
                'Longitude': lon,
                'HasCatch': 0,
                'CatchWeight': 0,
                **env_vars
            }
            all_data.append(record)
        
        # Close datasets to free memory
        if bio_data is not None:
            bio_data.close()
        if phy_data is not None:
            phy_data.close()
    
    return pd.DataFrame(all_data)


def load_and_filter_catch_data(csv_path, region_file, region_name, start_date='07-01-2019', end_date='11-30-2024'):
    """Load and filter catch data for the specified region and date range"""
    print(f"Loading catch data from: {csv_path}")
    
    # Load catch data
    df = pd.read_csv(csv_path, sep=';')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    
    # Get region boundaries
    region = get_region_border(region_file, region_name)
    
    # Filter by date range
    date_set = find_days_interval(start_date, end_date)
    df_filtered = df[df['Date'].isin(date_set)]
    
    # Filter by region
    df_filtered = df_filtered[
        ~((df_filtered["CatchLat"] < region["SW"]["lat"]) | 
          (df_filtered["CatchLat"] > region["NE"]["lat"]))
    ]
    df_filtered = df_filtered[
        ~((df_filtered["CatchLon"] < region["SW"]["lon"]) | 
          (df_filtered["CatchLon"] > region["NE"]["lon"]))
    ]
    
    print(f"Filtered data: {len(df_filtered)} records")
    print(f"Date range: {df_filtered['Date'].min()} to {df_filtered['Date'].max()}")
    print(f"Unique dates: {df_filtered['Date'].nunique()}")
    
    return df_filtered, region


def print_results_summary(df_result, output_path, pool_output_path=None):
    """Print a summary of the results"""
    print("\n" + "=" * 60)
    print("METHOD 1 RESULTS SUMMARY")
    print("=" * 60)
    
    if len(df_result) == 0:
        print("No records generated!")
        return
    
    print(f"Generated dataset with {len(df_result)} records")
    print(f"Catch/No-catch distribution: {df_result['HasCatch'].value_counts().to_dict()}")
    print(f"Date range: {df_result['Date'].min()} to {df_result['Date'].max()}")
    print(f"Unique dates: {df_result['Date'].nunique()}")
    print(f"Catch records: {(df_result['HasCatch'] == 1).sum()}")
    print(f"No-catch records: {(df_result['HasCatch'] == 0).sum()}")
    
    # Environmental data summary
    env_cols = [col for col in df_result.columns if 'bio_' in col or 'phy_' in col]
    print(f"Environmental columns: {len(env_cols)}")
    
    if len(env_cols) > 0:
        non_null_env = df_result[env_cols].notna().sum().sum()
        total_env_values = len(df_result) * len(env_cols)
        print(f"Non-null environmental values: {non_null_env}/{total_env_values} ({non_null_env/total_env_values*100:.1f}%)")
    
    print(f"Dataset saved to: {output_path}")
    if pool_output_path:
        print(f"Location pool saved to: {pool_output_path}")
    print("=" * 60)


def main():
    """Main function to run Method 1 implementation"""
    
    # Configuration
    CSV_PATH = '/home/anna/msc_oppgave/fish-forecast/TEST_MAC_ERS_VMS_COMPLETE.csv'
    REGION_FILE = '/home/anna/msc_oppgave/fish-forecast/areas.json'
    REGION_NAME = 'test'
    
    # Environmental data paths
    DATASET_PATH = '/home/anna/msc_oppgave/fish-forecast/Data/mac_winter_areas/'
    BIO_PATH = DATASET_PATH + 'bio'
    PHY_PATH = DATASET_PATH + 'phy'
    
    # Processing parameters
    MIN_DISTANCE = 0.09  # Minimum distance between catch and no-catch points
    POOL_SIZE = 200_000     # Size of pre-generated location pool
    
    # Dataset size options
    USE_FULL_DATASET = True
    USE_CUSTOM_SUBSET = not USE_FULL_DATASET
    CUSTOM_SUBSET_SIZE = 100  # Adjust as needed
    YEAR_SUBSET = [2019, 2024]  # Process data form 2019 to 2024  

    print("Method 1 Implementation - Fish Catch Correlation Analysis")
    print("=" * 60)
    
    try:
        # Load and filter catch data
        df_catch, region = load_and_filter_catch_data(
            CSV_PATH, REGION_FILE, REGION_NAME, 
            start_date=f'07-01-{YEAR_SUBSET[0]}', end_date=f'11-30-{YEAR_SUBSET[1]}'
        )
        
        # Select dataset size
        if USE_FULL_DATASET:
            print("Running on FULL DATASET - this may take a long time...")
            final_df = df_catch
            output_filename = 'correlation_dataset_method1_full_wVMS.csv'
        elif USE_CUSTOM_SUBSET:
            print(f"Running on CUSTOM SUBSET of {CUSTOM_SUBSET_SIZE} records...")
            final_df = df_catch.head(CUSTOM_SUBSET_SIZE)
            output_filename = f'correlation_dataset_method1_subset_{CUSTOM_SUBSET_SIZE}.csv'
        else:
            print("No dataset option selected. Using small test dataset...")
            final_df = df_catch.head(20)
            output_filename = 'correlation_dataset_method1_test.csv'
        
        print(f"Processing {len(final_df)} catch records")
        print(f"Date range: {final_df['Date'].min()} to {final_df['Date'].max()}")
        print(f"Unique dates: {final_df['Date'].nunique()}")
        
        # Run Method 1 implementation
        print("\nStarting Method 1 processing...")

        # Create pool output filename
        pool_output_filename = output_filename.replace('.csv', '_location_pool.csv')
        pool_output_path = f'/home/anna/msc_oppgave/fish-forecast/{pool_output_filename}'
        
        result_df = method1_implementation(
            final_df, region, BIO_PATH, PHY_PATH, 
            min_distance=MIN_DISTANCE, pool_size=POOL_SIZE,
            save_pool=True, pool_output_path=pool_output_path
        )
        
        # Save results
        output_path = f'/home/anna/msc_oppgave/fish-forecast/{output_filename}'
        if len(result_df) > 0:
            result_df.to_csv(output_path, index=False, sep=';')
        
        # Print summary
        print_results_summary(result_df, output_path, pool_output_path)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

