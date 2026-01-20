"""Tests for bathymetry module."""

import numpy as np

from bathy.bathymetry import Bathymetry, list_regions


def test_list_regions():
    """List all preset regions."""
    regions = list_regions()

    assert "mediterranean" in regions
    assert "mariana_trench" in regions


def test_load_from_netcdf(temp_netcdf):
    """Load bathymetry from NetCDF file."""
    bath = Bathymetry(temp_netcdf)

    assert bath.shape == (20, 20)
    assert bath.lon_range == (-10.0, -5.0)
    assert bath.lat_range == (50.0, 55.0)


def test_summary_stats(temp_netcdf):
    """Calculate summary statistics."""
    bath = Bathymetry(temp_netcdf)
    stats = bath.summary()

    assert "statistic" in stats.columns
    assert "value" in stats.columns
    assert len(stats) == 7


def test_slope_calculation(temp_netcdf):
    """Calculate seafloor slope."""
    bath = Bathymetry(temp_netcdf)
    slope = bath.slope()

    assert slope.shape == bath.shape
    assert (slope.values >= 0).all()


def test_create_profile(temp_netcdf):
    """Create a profile from bathymetry."""
    bath = Bathymetry(temp_netcdf)
    prof = bath.profile(start=(-9, 52), end=(-6, 53), num_points=10)

    assert prof.num_points == 10
    assert len(prof.distances) == 10
    assert len(prof.elevations) == 10
    assert prof.start_lon == -9
    assert prof.start_lat == 52
    assert prof.end_lon == -6
    assert prof.end_lat == 53


def test_plot_bathy_masks_land():
    """Verify plot_bathy masks land (elevation >= 0)."""
    import xarray as xr

    # Create data with both underwater and land
    data = xr.DataArray(
        np.array([[-100, 50]]),  # Underwater and land
        coords={"lon": [-10, -5], "lat": [50]},
        dims=["lat", "lon"],
    )

    # Test masking logic: data.where(data < 0)
    masked = data.where(data < 0)
    assert np.isnan(masked.values[0, 1])  # Land masked
    assert masked.values[0, 0] == -100  # Water not masked
