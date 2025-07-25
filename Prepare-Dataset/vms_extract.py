"""
Script for finding VMS records for vessels that have reported catches in the ERS data.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

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
    date_format = find_format(year)
    df_vms['Tidspunkt (UTC)'] = pd.to_datetime(
        df_vms['Tidspunkt (UTC)'],
        format=date_format,
        errors="coerce",
        dayfirst=True
    )

    df_vms_subset = df_vms[df_vms['Fartøyidentifikasjon']
                           .isin(df_ers['Fartøyidentifikasjon'].dropna())]
    print(f"Number of VMS records: {df_vms_subset.shape[0]}")

    # filter out locations that are not in the time range of the ERS data
    df_ers['Starttidspunkt'] = pd.to_datetime(df_ers['Starttidspunkt'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
    df_ers['Stopptidspunkt'] = pd.to_datetime(df_ers['Stopptidspunkt'], format='%d.%m.%Y %H:%M:%S', errors='coerce')

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

    print("ERS:" + str(len(df_ers['Fartøyidentifikasjon'].unique())) + " unique vessels")
    print("VMS:" + str(len(df_valid_vessels['Fartøyidentifikasjon'].unique())) + " unique vessels")
    print(f"Missing VMS vessels: {set(df_ers['Fartøyidentifikasjon'].unique()) - set(df_valid_vessels['Fartøyidentifikasjon'].unique())}")

    return df_valid_vessels


if __name__ == "__main__":

    ERS_PATH = '/home/anna/msc_oppgave/fish-forecast/All_MAC_ERS.csv'
    df_ers = pd.read_csv(ERS_PATH, sep=';')
    df_ers['Source'] = 'ERS'

    all_mac_ERS_VMS = df_ers.copy()

    for year in range(2011, 2025):
        VMS_PATH = f'/home/anna/msc_oppgave/data/VMS/{year}-VMS.csv'
        df_vms = pd.read_csv(VMS_PATH, sep=';', low_memory=False)
        df_vms['Source'] = 'VMS'

        df_vms['Fartøyidentifikasjon'] = (df_vms['Fartøynavn'].astype(str) + " - " + 
                                        df_vms['Radiokallesignal'].astype(str) + " - " + 
                                        df_vms['Registreringsmerke'].astype(str))
        # # Isolate rows from the ERS data that are relevant for the current year
        # df = df_ers[df_ers['Date'].dt.year == year]

        df_vms = find_vms_records(df_ers, df_vms, year)
        df_vms.to_csv(f"filtered_vms_ALL-MAC_{year}.csv", sep=';', index=False)
        del df_vms

    #     combined_df = pd.concat([all_mac_ERS_VMS, df_vms], ignore_index=True)
    # combined_df.to_csv(f"All_MAC_ERS_VMS.csv", sep=';', index=False)