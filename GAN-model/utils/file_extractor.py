import pathlib
from typing import Set, Union

def extract_date_from_filename(file_path: Union[str, pathlib.Path]) -> str:
    """
    Extract the date part from a filename.
    Assumes filenames have the form: <prefix><date>.ext
    e.g. "chl_2023-06-15.png" or "sal_h0_2023-06-15.png".
    also handles cases where the filename is just a date like "2023-06-15.png".
    Returns the date part as a string.
    """
    if '_' not in file_path:
        return pathlib.Path(file_path).stem
    else:
        return pathlib.Path(file_path).stem.split('_')[-1]


def get_available_dates(dir_path: Union[str, pathlib.Path], extension: str = "png") -> Set[str]:
    """
    Return a set of date strings (extracted via extract_date_from_filename) for all files
    matching *.{extension} in dir_path.
    """
    p = pathlib.Path(dir_path)
    return {extract_date_from_filename(f.name) for f in p.glob(f"*.{extension}")}
