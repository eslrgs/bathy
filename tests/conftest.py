"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def fake_data():
    """Simple fake bathymetry data."""
    lons = np.linspace(-10, -5, 20)
    lats = np.linspace(50, 55, 20)
    depths = np.random.rand(20, 20) * -100

    return xr.DataArray(
        depths,
        coords={"lon": lons, "lat": lats},
        dims=["lat", "lon"],
        name="elevation",
    )


@pytest.fixture
def temp_netcdf(fake_data):
    """Temporary NetCDF file with fake data."""
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        fake_data.to_netcdf(tmp_path)

    yield tmp_path

    tmp_path.unlink()
