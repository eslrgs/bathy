"""Tests for analysis module."""

import geopandas as gpd
import numpy as np
import polars as pl
import pytest
import xarray as xr

from bathy.analysis import (
    _cell_size_metres,
    aspect,
    bpi,
    contours,
    curvature,
    geomorphons,
    hypsometric_curve,
    hypsometric_index,
    rugosity,
    slope,
    summary,
)


def test_summary_stats(fake_bathy):
    """Calculate summary statistics."""
    result = summary(fake_bathy)

    assert "statistic" in result.columns
    assert "value" in result.columns
    assert len(result) == 8


def test_slope_calculation(fake_bathy):
    """Calculate seafloor slope."""
    slope_da = slope(fake_bathy)

    assert slope_da.shape == fake_bathy.shape
    assert (slope_da.values >= 0).all()


# Hypsometry tests


def test_hypsometric_index_range(fake_bathy):
    """Hypsometric index should be between 0 and 1."""
    hi = hypsometric_index(fake_bathy)

    assert 0 <= hi <= 1


def test_hypsometric_index_uniform_distribution(uniform_bathy):
    """Uniform distribution should have HI close to 0.5."""
    hi = hypsometric_index(uniform_bathy)

    assert abs(hi - 0.5) < 0.01


def test_hypsometric_index_convex(convex_bathy):
    """Convex distribution (more high values) should have HI > 0.5."""
    hi = hypsometric_index(convex_bathy)

    assert hi > 0.5


def test_hypsometric_index_flat_surface(flat_bathy):
    """Flat surface (constant elevation) should return NaN."""
    hi = hypsometric_index(flat_bathy)

    assert np.isnan(hi)


def test_hypsometric_curve(fake_bathy):
    """Hypsometric curve returns normalised, monotonic DataFrame."""
    df = hypsometric_curve(fake_bathy, bins=50)

    assert isinstance(df, pl.DataFrame)
    assert set(df.columns) == {"relative_area", "relative_elevation"}
    assert len(df) == 50

    rel_area = df["relative_area"].to_numpy()
    rel_elev = df["relative_elevation"].to_numpy()

    assert 0 <= rel_area.min() and rel_area.max() <= 1
    assert 0 <= rel_elev.min() and rel_elev.max() <= 1

    assert np.all(np.diff(rel_area) <= 0)


def test_curvature_calculation(fake_bathy):
    """Calculate seafloor curvature."""
    curv = curvature(fake_bathy)

    assert curv.shape == fake_bathy.shape
    assert curv.name == "curvature"


def test_bpi_calculation(fake_bathy):
    """Calculate Bathymetric Position Index."""
    bpi_da = bpi(fake_bathy, radius_km=1.0)

    assert bpi_da.shape == fake_bathy.shape
    assert bpi_da.name == "bpi"


def test_bpi_flat_surface_is_zero(flat_bathy):
    """Flat surface should have BPI ≈ 0 everywhere."""
    bpi_da = bpi(flat_bathy, radius_km=1.0)

    assert np.allclose(bpi_da.values, 0, atol=1e-10)


def test_bpi_peak_is_positive():
    """A peak (high point surrounded by low) should have positive BPI."""
    elevations = np.full((21, 21), -1000.0)
    elevations[10, 10] = -500.0

    data = xr.DataArray(
        elevations,
        coords={"lon": np.linspace(-10, -5, 21), "lat": np.linspace(50, 55, 21)},
        dims=["lat", "lon"],
    )
    bpi_da = bpi(data, radius_km=50)

    assert bpi_da.values[10, 10] > 0


def test_bpi_valley_is_negative():
    """A valley (low point surrounded by high) should have negative BPI."""
    elevations = np.full((21, 21), -500.0)
    elevations[10, 10] = -1000.0

    data = xr.DataArray(
        elevations,
        coords={"lon": np.linspace(-10, -5, 21), "lat": np.linspace(50, 55, 21)},
        dims=["lat", "lon"],
    )
    bpi_da = bpi(data, radius_km=50)

    assert bpi_da.values[10, 10] < 0


def test_rugosity_calculation(fake_bathy):
    """Calculate Vector Ruggedness Measure."""
    rug = rugosity(fake_bathy, radius_km=1.0)

    assert rug.shape == fake_bathy.shape
    assert rug.name == "rugosity"


def test_rugosity_range(fake_bathy):
    """VRM values should be in [0, 1]."""
    rug = rugosity(fake_bathy, radius_km=1.0)

    assert rug.values.min() >= 0
    assert rug.values.max() <= 1


def test_rugosity_flat_surface_is_zero(flat_bathy):
    """Flat surface should have VRM ≈ 0 everywhere."""
    rug = rugosity(flat_bathy, radius_km=1.0)

    assert np.allclose(rug.values, 0, atol=1e-10)


def test_rugosity_tilted_plane_is_zero():
    """Uniformly sloping surface should have VRM ≈ 0 (all normals parallel)."""
    x = np.linspace(0, 20, 30)
    y = np.linspace(0, 20, 30)
    xx, _ = np.meshgrid(x, y)
    ramp = xr.DataArray(
        -xx * 50.0,
        coords={"lon": np.linspace(-10, -5, 30), "lat": np.linspace(50, 55, 30)},
        dims=["lat", "lon"],
    )

    assert np.allclose(rugosity(ramp).values, 0, atol=1e-6)


def test_rugosity_rough_exceeds_flat(flat_bathy):
    """Rough terrain should have higher mean VRM than flat terrain."""
    rng = np.random.default_rng(0)
    rough_data = xr.DataArray(
        rng.uniform(-1000, 0, (20, 20)),
        coords={"lon": np.linspace(-10, -5, 20), "lat": np.linspace(50, 55, 20)},
        dims=["lat", "lon"],
    )

    assert rugosity(rough_data).values.mean() > rugosity(flat_bathy).values.mean()


def test_aspect_calculation(fake_bathy):
    """Calculate seafloor aspect."""
    asp = aspect(fake_bathy)

    assert asp.shape == fake_bathy.shape
    assert asp.name == "aspect"


def test_aspect_range(fake_bathy):
    """Aspect values should be in [0, 360)."""
    asp = aspect(fake_bathy)
    valid = asp.values[~np.isnan(asp.values)]

    assert valid.min() >= 0
    assert valid.max() < 360


def test_aspect_flat_surface_is_nan(flat_bathy):
    """Flat surface (zero gradient) should return NaN everywhere."""
    asp = aspect(flat_bathy)

    assert np.all(np.isnan(asp.values))


def test_aspect_south_facing():
    """Surface ascending northward faces south (downslope = 180°)."""
    lats = np.linspace(50, 55, 20)
    elevations = np.outer(np.linspace(-1000, -500, 20), np.ones(20))
    data = xr.DataArray(
        elevations,
        coords={"lon": np.linspace(-10, -5, 20), "lat": lats},
        dims=["lat", "lon"],
    )
    asp = aspect(data)

    assert np.allclose(asp.values[1:-1, 1:-1], 180, atol=1e-6)


def test_aspect_west_facing():
    """Surface ascending eastward faces west (downslope = 270°)."""
    elevations = np.outer(np.ones(20), np.linspace(-1000, -500, 20))
    data = xr.DataArray(
        elevations,
        coords={"lon": np.linspace(-10, -5, 20), "lat": np.linspace(50, 55, 20)},
        dims=["lat", "lon"],
    )
    asp = aspect(data)

    assert np.allclose(asp.values[1:-1, 1:-1], 270, atol=1e-6)


def test_geomorphons_shape_and_name(fake_bathy):
    """geomorphons returns correct shape and DataArray name."""
    geom = geomorphons(fake_bathy, lookup_km=1.0)

    assert geom.shape == fake_bathy.shape
    assert geom.name == "geomorphons"


def test_geomorphons_classes_in_range(fake_bathy):
    """All class codes should be integers in 1–10."""
    geom = geomorphons(fake_bathy, lookup_km=1.0)

    assert geom.values.min() >= 1
    assert geom.values.max() <= 10


def test_geomorphons_flat_surface_is_flat(flat_bathy):
    """Flat surface should be classified entirely as flat (class 1)."""
    geom = geomorphons(flat_bathy, lookup_km=1.0)

    assert np.all(geom.values[2:-2, 2:-2] == 1)


def test_geomorphons_peak_classified_correctly():
    """Isolated high point surrounded by deep flat should be classified as peak."""
    elevations = np.full((21, 21), -1000.0)
    elevations[10, 10] = -100.0

    data = xr.DataArray(
        elevations,
        coords={"lon": np.linspace(-10, -5, 21), "lat": np.linspace(50, 55, 21)},
        dims=["lat", "lon"],
    )
    geom = geomorphons(data, lookup_km=1.0)

    assert geom.values[10, 10] == 2


def test_geomorphons_pit_classified_correctly():
    """Isolated deep point surrounded by shallow flat should be classified as pit."""
    elevations = np.full((21, 21), -100.0)
    elevations[10, 10] = -1000.0

    data = xr.DataArray(
        elevations,
        coords={"lon": np.linspace(-10, -5, 21), "lat": np.linspace(50, 55, 21)},
        dims=["lat", "lon"],
    )
    geom = geomorphons(data, lookup_km=1.0)

    assert geom.values[10, 10] == 10


def test_geomorphons_ridge_behind_valley():
    """A cell between a valley and a ridge should not be classified as flat.

    This exercises the line-of-sight scan: the ridge at the lookup
    distance is hidden behind the nearby valley when only the endpoint
    is checked, but the full scan detects both features.
    """
    elevations = np.full((21, 21), -500.0)
    # Valley one step east of centre
    elevations[10, 11] = -900.0
    # Ridge at the lookup boundary further east
    elevations[10, 14] = -100.0

    data = xr.DataArray(
        elevations,
        coords={"lon": np.linspace(-10, -5, 21), "lat": np.linspace(50, 55, 21)},
        dims=["lat", "lon"],
    )
    geom = geomorphons(data, lookup_km=50.0)

    # Centre cell sees both higher and lower terrain to the east,
    # so it must not be classified as flat (1) or peak (2) or pit (10).
    assert geom.values[10, 10] not in (1, 2, 10)


# Projected CRS tests


def test_cell_size_projected(fake_projected_data):
    """Projected data returns cell sizes directly from coordinate spacing."""
    dy, dx = _cell_size_metres(fake_projected_data)
    expected = 10000 / 19  # 10 km range / 19 intervals
    assert abs(dx - expected) < 1
    assert abs(dy - expected) < 1


def test_slope_projected(fake_projected_data):
    """Slope works on projected data."""
    slope_da = slope(fake_projected_data)
    assert slope_da.shape == fake_projected_data.shape
    assert (slope_da.values >= 0).all()


@pytest.mark.parametrize("radius", [-1.0, 0.0])
def test_bpi_invalid_radius(fake_bathy, radius):
    """Non-positive radius_km raises ValueError."""
    with pytest.raises(ValueError, match="radius_km must be positive"):
        bpi(fake_bathy, radius_km=radius)


@pytest.mark.parametrize("radius", [-1.0, 0.0])
def test_rugosity_invalid_radius(fake_bathy, radius):
    """Non-positive radius_km raises ValueError."""
    with pytest.raises(ValueError, match="radius_km must be positive"):
        rugosity(fake_bathy, radius_km=radius)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lookup_km": -1.0},
        {"lookup_km": 0.0},
        {"flatness_threshold": -1.0},
        {"flatness_threshold": 0.0},
    ],
)
def test_geomorphons_invalid_params(fake_bathy, kwargs):
    """Non-positive lookup_km or flatness_threshold raises ValueError."""
    with pytest.raises(ValueError):
        geomorphons(fake_bathy, **kwargs)


@pytest.mark.parametrize("bins", [-1, 0])
def test_hypsometric_curve_invalid_bins(fake_bathy, bins):
    """Non-positive bins raises ValueError."""
    with pytest.raises(ValueError, match="bins must be positive"):
        hypsometric_curve(fake_bathy, bins=bins)


def test_contours_returns_geodataframe(fake_bathy):
    """contours returns a GeoDataFrame with depth and geometry columns."""
    gdf = contours(fake_bathy, levels=[-75, -50, -25])

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert "depth" in gdf.columns
    assert "geometry" in gdf.columns
    assert len(gdf) > 0


def test_contours_levels_match(fake_bathy):
    """Returned depths should be a subset of the requested levels."""
    levels = [-80, -60, -40, -20]
    gdf = contours(fake_bathy, levels=levels)

    assert set(gdf["depth"].unique()).issubset(set(levels))


def test_contours_with_interval(fake_bathy):
    """contours with interval produces regularly spaced levels."""
    gdf = contours(fake_bathy, interval=25)

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) > 0
    depths = sorted(gdf["depth"].unique())
    if len(depths) > 1:
        spacings = np.diff(depths)
        assert np.allclose(spacings, 25)


def test_contours_invalid_interval(fake_bathy):
    """Non-positive interval raises ValueError."""
    with pytest.raises(ValueError, match="interval must be positive"):
        contours(fake_bathy, interval=-10)


def test_contours_flat_surface(flat_bathy):
    """Flat surface produces no contour lines."""
    gdf = contours(flat_bathy, levels=[-400, -300])

    assert len(gdf) == 0
