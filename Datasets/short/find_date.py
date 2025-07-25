import os

def get_dates_from_filenames(folder, prefix='', extensions={'.png', '.jpg', '.jpeg', '.tif'}):
    return {
        os.path.splitext(f)[0].replace(prefix, '')
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and f.startswith(prefix)
        and os.path.splitext(f)[1].lower() in extensions
    }

def find_unique_dates(folder1, folder2, prefix1='', prefix2=''):
    dates1 = get_dates_from_filenames(folder1, prefix1)
    dates2 = get_dates_from_filenames(folder2, prefix2)

    unique_dates = dates1.symmetric_difference(dates2)
    return sorted(unique_dates)

if __name__ == "__main__":
    folder1 = '/home/anna/msc_oppgave/fish-forecast/Datasets/all_catch_small/catch'
    folder2 = '/home/anna/msc_oppgave/fish-forecast/Datasets/all_catch_small/no3/29.0'

    # Match prefix of each folder
    prefix1 = 'catch_'
    prefix2 = ''

    unique_dates = find_unique_dates(folder1, folder2, prefix1, prefix2)

    if unique_dates:
        print("Dates not present in both folders:")
        for date in unique_dates:
            print(f"  {date}")
    else:
        print("All dates are present in both folders.")
