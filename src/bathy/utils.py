"""Utility functions for the bathy package."""

import xarray as xr


def get_extent(data: xr.DataArray) -> list[float]:
    """
    Get extent for matplotlib imshow.

    Parameters
    ----------
    data : xr.DataArray
        Data array with lon and lat coordinates

    Returns
    -------
    list[float]
        Extent as [lon_min, lon_max, lat_min, lat_max]
    """
    return [float(data.lon.min()), float(data.lon.max()), float(data.lat.min()), float(data.lat.max())]
