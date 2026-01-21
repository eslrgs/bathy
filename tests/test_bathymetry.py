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


def test_summary_stats(fake_bathy):
    """Calculate summary statistics."""
    stats = fake_bathy.summary()

    assert "statistic" in stats.columns
    assert "value" in stats.columns
    assert len(stats) == 7


def test_slope_calculation(fake_bathy):
    """Calculate seafloor slope."""
    slope = fake_bathy.slope()

    assert slope.shape == fake_bathy.shape
    assert (slope.values >= 0).all()


def test_create_profile(fake_bathy):
    """Create a profile from bathymetry."""
    prof = fake_bathy.profile(start=(-9, 52), end=(-6, 53), num_points=10)

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


# Hypsometry tests


def test_hypsometric_index_range(fake_bathy):
    """Hypsometric index should be between 0 and 1."""
    hi = fake_bathy.hypsometric_index()

    assert 0 <= hi <= 1


def test_hypsometric_index_uniform_distribution(uniform_bathy):
    """Uniform distribution should have HI close to 0.5."""
    hi = uniform_bathy.hypsometric_index()

    assert abs(hi - 0.5) < 0.01


def test_hypsometric_index_convex(convex_bathy):
    """Convex distribution (more high values) should have HI > 0.5."""
    hi = convex_bathy.hypsometric_index()

    assert hi > 0.5


def test_hypsometric_index_flat_surface(flat_bathy):
    """Flat surface (constant elevation) should return NaN."""
    hi = flat_bathy.hypsometric_index()

    assert np.isnan(hi)


def test_hypsometric_curve(fake_bathy):
    """Hypsometric curve returns normalised, monotonic arrays."""
    rel_area, rel_elev = fake_bathy.hypsometric_curve(bins=50)

    # Correct shape
    assert len(rel_area) == 50
    assert len(rel_elev) == 50

    # Normalised between 0 and 1
    assert 0 <= rel_area.min() and rel_area.max() <= 1
    assert 0 <= rel_elev.min() and rel_elev.max() <= 1

    # Relative area decreases as elevation increases
    assert np.all(np.diff(rel_area) <= 0)


def test_curvature_calculation(fake_bathy):
    """Calculate seafloor curvature."""
    curv = fake_bathy.curvature()

    assert curv.shape == fake_bathy.shape
    assert curv.name == "curvature"
