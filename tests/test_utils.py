"""Tests for CRS detection and coordinate utilities."""

import numpy as np
import pytest
import xarray as xr

from bathy.utils import axis_labels, get_crs, get_dim_names, is_geographic, is_projected


def test_get_crs_geographic(fake_data):
    """No CRS attached returns None."""
    assert get_crs(fake_data) is None


def test_get_crs_projected(fake_projected_data):
    """Projected CRS is detected."""
    crs = get_crs(fake_projected_data)
    assert crs is not None
    assert crs.to_epsg() == 32629


def test_is_geographic_no_crs(fake_data):
    """No CRS defaults to geographic."""
    assert is_geographic(fake_data) is True
    assert is_projected(fake_data) is False


def test_is_projected(fake_projected_data):
    """Projected CRS is correctly identified."""
    assert is_projected(fake_projected_data) is True
    assert is_geographic(fake_projected_data) is False


def test_get_dim_names_geographic(fake_data):
    """Geographic data has lon/lat dims."""
    assert get_dim_names(fake_data) == ("lon", "lat")


def test_get_dim_names_projected(fake_projected_data):
    """Projected data has x/y dims."""
    assert get_dim_names(fake_projected_data) == ("x", "y")


def test_get_dim_names_invalid():
    """Missing expected dims raises ValueError."""
    da = xr.DataArray(
        np.zeros((5, 5)),
        coords={"a": range(5), "b": range(5)},
        dims=["a", "b"],
    )
    with pytest.raises(ValueError, match="Expected dimensions"):
        get_dim_names(da)


def test_axis_labels_geographic(fake_data):
    assert axis_labels(fake_data) == ("Longitude (°)", "Latitude (°)")


def test_axis_labels_projected(fake_projected_data):
    assert axis_labels(fake_projected_data) == ("Easting (m)", "Northing (m)")
