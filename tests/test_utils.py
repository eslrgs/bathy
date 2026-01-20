"""Tests for utils module."""

import numpy as np
import xarray as xr

from bathy.utils import get_extent


def test_get_extent():
    """Test get_extent returns [lon_min, lon_max, lat_min, lat_max]."""
    data = xr.DataArray(
        np.zeros((3, 3)),
        coords={"lon": [0.0, 1.0, 2.0], "lat": [50.0, 51.0, 52.0]},
        dims=["lat", "lon"],
    )

    assert get_extent(data) == [0.0, 2.0, 50.0, 52.0]
