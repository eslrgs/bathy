"""Sample datasets for examples and testing."""

import pooch
import xarray as xr

_BASE_URL = "https://github.com/eslrgs/bathy/releases/download/v0.1.0/"


def sample_data() -> xr.DataArray:
    """
    Load the NE Atlantic sample bathymetry dataset.

    Downloads a GEBCO 2025 extract covering the Celtic Sea and Bay of Biscay
    (lon: -12.08 to -5.23, lat: 46.05 to 49.54) on first call, then uses
    the cached copy.

    The data is derived from the GEBCO 2025 Grid:

        GEBCO Compilation Group (2025) GEBCO 2025 Grid
        (doi:10.5285/37c52e96-24ea-67ce-e063-7086abc05f29)

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
        known_hash="e153e0ce4a4cc72e32aa2595aea322c0f247a57e5d173bd020fe3d77b7d827fb",
    )
    ds = xr.open_dataset(path)
    return ds["elevation"]
