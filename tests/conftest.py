"""Pytest configuration and shared fixtures."""

import os
import tempfile

import numpy as np
import pytest
import xarray as xr

from bathy import Bathymetry


def _make_bathy(elevations, n=20):
    """Create Bathymetry from elevation array."""
    data = xr.DataArray(
        elevations,
        coords={"lon": np.linspace(-10, -5, n), "lat": np.linspace(50, 55, n)},
        dims=["lat", "lon"],
        name="elevation",
    )
    return Bathymetry.from_array(data)


@pytest.fixture
def fake_data():
    """Raw DataArray with random bathymetry data."""
    return xr.DataArray(
        np.random.rand(20, 20) * -100,
        coords={"lon": np.linspace(-10, -5, 20), "lat": np.linspace(50, 55, 20)},
        dims=["lat", "lon"],
        name="elevation",
    )


@pytest.fixture
def fake_bathy(fake_data):
    """Bathymetry with random data."""
    return Bathymetry.from_array(fake_data)


@pytest.fixture
def uniform_bathy():
    """Bathymetry with uniform distribution (HI ~ 0.5)."""
    return _make_bathy(np.linspace(-1000, 0, 10000).reshape(100, 100), n=100)


@pytest.fixture
def convex_bathy():
    """Bathymetry with convex distribution (HI > 0.5)."""
    elevations = -np.abs(np.random.default_rng(42).normal(0, 100, (50, 50)))
    return _make_bathy(elevations, n=50)


@pytest.fixture
def flat_bathy():
    """Bathymetry with flat surface (HI = NaN)."""
    return _make_bathy(np.full((10, 10), -500.0), n=10)


@pytest.fixture
def temp_netcdf(fake_bathy):
    """Temporary NetCDF file for testing file loading."""
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        fake_bathy.data.to_netcdf(tmp.name)
        path = tmp.name
    yield path
    os.unlink(path)
