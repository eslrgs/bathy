"""Sample datasets for examples and testing."""

import pooch
import xarray as xr

# TODO: Upload when ready to release v.0.1.0
_BASE_URL = "https://github.com/eslrgs/bathy/releases/download/v0.1.0/"


def sample_data() -> xr.DataArray:
    """
    Load the NE Atlantic sample bathymetry dataset.

    Downloads a GEBCO 2025 extract covering the Celtic Sea and Bay of Biscay
    (lon: -12.08 to -5.23, lat: 46.05 to 49.54) on first call, then uses
    the cached copy.

    Returns
    -------
    xr.DataArray
        Elevation data with 'lon' and 'lat' coordinates.

    Examples
    --------
    >>> import bathy
    >>> data = bathy.sample_data()
    """
    path = pooch.retrieve(
        url=f"{_BASE_URL}ne_atlantic_gebco.nc",
        known_hash=None,
    )
    ds = xr.open_dataset(path)
    return ds["elevation"]
