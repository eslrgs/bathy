"""Pytest configuration and shared fixtures."""

import os
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import rioxarray  # noqa: F401, E402
import xarray as xr  # noqa: E402

from bathy.profile import extract_profile  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all figures after each test to free memory."""
    yield
    plt.close("all")


def _make_bathy(elevations, n=20):
    """Create a DataArray from an elevation array."""
    return xr.DataArray(
        elevations,
        coords={"lon": np.linspace(-10, -5, n), "lat": np.linspace(50, 55, n)},
        dims=["lat", "lon"],
        name="elevation",
    )


@pytest.fixture
def fake_data():
    """Raw DataArray with random bathymetry data."""
    return xr.DataArray(
        np.random.default_rng(42).random((20, 20)) * -100,
        coords={"lon": np.linspace(-10, -5, 20), "lat": np.linspace(50, 55, 20)},
        dims=["lat", "lon"],
        name="elevation",
    )


@pytest.fixture
def fake_bathy(fake_data):
    """DataArray alias used in bathymetry tests."""
    return fake_data


@pytest.fixture
def uniform_bathy():
    """DataArray with uniform distribution (HI ~ 0.5)."""
    return _make_bathy(np.linspace(-1000, 0, 10000).reshape(100, 100), n=100)


@pytest.fixture
def convex_bathy():
    """DataArray with convex distribution (HI > 0.5)."""
    elevations = -np.abs(np.random.default_rng(42).normal(0, 100, (50, 50)))
    return _make_bathy(elevations, n=50)


@pytest.fixture
def flat_bathy():
    """DataArray with flat surface (HI = NaN)."""
    return _make_bathy(np.full((10, 10), -500.0), n=10)


@pytest.fixture
def fake_profile(fake_data):
    """Straight-line profile across the fake bathymetry grid."""
    return extract_profile(
        fake_data, start=(-9, 52), end=(-6, 53), num_points=20, name="Test Profile"
    )


@pytest.fixture
def fake_projected_data():
    """DataArray with UTM-like projected coordinates (EPSG:32629)."""
    da = xr.DataArray(
        np.random.default_rng(42).random((20, 20)) * -100,
        coords={
            "x": np.linspace(500000, 510000, 20),
            "y": np.linspace(5500000, 5510000, 20),
        },
        dims=["y", "x"],
        name="elevation",
    )
    da = da.rio.write_crs("EPSG:32629")
    return da


@pytest.fixture
def temp_netcdf(fake_bathy):
    """Temporary NetCDF file for testing file loading."""
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        fake_bathy.to_netcdf(tmp.name)
        path = tmp.name
    yield path
    os.unlink(path)


@pytest.fixture
def temp_geotiff(fake_bathy):
    """Temporary GeoTIFF file for testing file loading."""
    da = fake_bathy.rename({"lon": "x", "lat": "y"})
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        da.rio.to_raster(tmp.name)
        path = tmp.name
    yield path
    os.unlink(path)
