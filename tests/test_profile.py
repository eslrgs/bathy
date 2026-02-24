"""Tests for profile module."""

import geopandas
import numpy as np
import pytest
from shapely.geometry import LineString

from bathy.profile import (
    _ensure_descending,
    extract_profile,
    knickpoints,
    profiles_from_gdf,
    to_gdf,
)


def test_create_profile(fake_data):
    """Create a profile between two points."""
    prof = extract_profile(
        fake_data, start=(-9, 52), end=(-6, 53), num_points=10, name="Test"
    )

    assert prof.name == "Test"
    assert len(prof.distances) == 10
    assert len(prof.elevations) == 10
    assert prof.start_lon == -9
    assert prof.start_lat == 52
    assert prof.end_lon == -6
    assert prof.end_lat == 53


def test_max_depth(fake_data):
    """Find the deepest point in a profile."""
    from bathy.profile import max_depth

    prof = extract_profile(fake_data, start=(-9, 52), end=(-6, 53), num_points=10)
    distance, depth = max_depth(prof)

    assert distance >= 0
    assert depth < 0


def test_gradient(fake_data):
    """Calculate profile gradient."""
    from bathy.profile import gradient

    prof = extract_profile(fake_data, start=(-9, 52), end=(-6, 53), num_points=10)
    grad = gradient(prof)

    assert len(grad) == 10
    assert isinstance(grad, np.ndarray)


def test_distance_axis_starts_at_start_point(fake_data):
    """Verify distance axis starts at zero for user-defined start point."""
    prof = extract_profile(fake_data, start=(-9, 52), end=(-6, 53), num_points=10)

    assert prof.distances[0] == 0
    assert np.all(np.diff(prof.distances) > 0)


def test_ensure_descending_bathymetry():
    """Test _ensure_descending with bathymetric (negative) elevations."""
    distances = np.array([0, 10, 20, 30])

    # Case 1: Already descending (shallow to deep: -100 to -4000)
    elevations_desc = np.array([-100, -1000, -2000, -4000])
    dist_out, elev_out = _ensure_descending(distances, elevations_desc)
    assert np.array_equal(dist_out, distances)
    assert np.array_equal(elev_out, elevations_desc)

    # Case 2: Ascending (deep to shallow: -4000 to -100), should flip and re-zero
    elevations_asc = np.array([-4000, -2000, -1000, -100])
    dist_out, elev_out = _ensure_descending(distances, elevations_asc)
    assert dist_out[0] == 0
    assert np.allclose(np.diff(dist_out), 10)
    assert np.array_equal(elev_out, elevations_asc[::-1])


def test_ensure_descending_topography():
    """Test _ensure_descending with topographic (positive) elevations."""
    distances = np.array([0, 10, 20, 30])

    # Case 1: Already descending (high to low: 1000 to 100)
    elevations_desc = np.array([1000, 500, 300, 100])
    dist_out, elev_out = _ensure_descending(distances, elevations_desc)
    assert np.array_equal(dist_out, distances)
    assert np.array_equal(elev_out, elevations_desc)

    # Case 2: Ascending (low to high: 100 to 1000), should flip and re-zero
    elevations_asc = np.array([100, 300, 500, 1000])
    dist_out, elev_out = _ensure_descending(distances, elevations_asc)
    assert dist_out[0] == 0
    assert np.allclose(np.diff(dist_out), 10)
    assert np.array_equal(elev_out, elevations_asc[::-1])


def test_knickpoints_returns_dataframe(fake_data):
    """Knickpoints returns DataFrame with expected columns."""
    prof = extract_profile(fake_data, start=(-9, 52), end=(-6, 53), num_points=50)
    kp = knickpoints(prof)

    assert hasattr(kp, "columns")
    assert set(kp.columns) == {"distance_km", "depth_m", "slope_break"}


def test_knickpoints_with_threshold(fake_data):
    """Higher threshold returns fewer knickpoints."""
    prof = extract_profile(fake_data, start=(-9, 52), end=(-6, 53), num_points=50)

    kp_low = knickpoints(prof, threshold=0.1)
    kp_high = knickpoints(prof, threshold=100)

    assert len(kp_high) <= len(kp_low)


def test_knickpoints_with_smoothing(fake_data):
    """Smoothing parameter is accepted."""
    prof = extract_profile(fake_data, start=(-9, 52), end=(-6, 53), num_points=50)
    kp = knickpoints(prof, smooth=3)

    assert set(kp.columns) == {"distance_km", "depth_m", "slope_break"}


# GeoDataFrame methods


def test_to_gdf(fake_profile):
    """to_gdf returns a GeoDataFrame with correct CRS, geometry, and statistics."""
    gdf = to_gdf(fake_profile)

    assert gdf.crs.to_epsg() == 4326
    assert gdf.geometry.iloc[0].geom_type == "LineString"
    coords = list(gdf.geometry.iloc[0].coords)
    assert coords[0] == pytest.approx((-9.0, 52.0))
    assert coords[-1] == pytest.approx((-6.0, 53.0))
    assert gdf["name"].iloc[0] == "Test Profile"
    assert gdf["total_distance_km"].iloc[0] > 0
    assert gdf["min_elevation_m"].iloc[0] <= gdf["max_elevation_m"].iloc[0]
    assert gdf["mean_elevation_m"].iloc[0] < 0


def test_to_gdf_metadata(fake_data):
    """to_gdf includes scalar metadata; core columns win over clashing metadata keys."""
    prof = extract_profile(
        fake_data,
        start=(-9, 52),
        end=(-6, 53),
        num_points=10,
        name="Real",
        metadata={"source": "survey_A", "name": "Impostor"},
    )
    gdf = to_gdf(prof)

    assert gdf["source"].iloc[0] == "survey_A"
    assert gdf["name"].iloc[0] == "Real"


def test_from_gdf_creates_profiles(fake_data):
    """profiles_from_gdf creates one Profile per LineString row."""
    gdf = geopandas.GeoDataFrame(
        {"label": ["A", "B"]},
        geometry=[
            LineString([(-9, 52), (-8, 52), (-7, 52)]),
            LineString([(-9, 53), (-8, 53), (-7, 53)]),
        ],
        crs="EPSG:4326",
    )
    profs = profiles_from_gdf(fake_data, gdf, id_column="label")

    assert len(profs) == 2
    assert profs[0].name == "A"
    assert profs[1].name == "B"


def test_from_gdf_sequential_naming_without_id_column(fake_data):
    """profiles_from_gdf uses sequential Feature_N names when id_column is None."""
    gdf = geopandas.GeoDataFrame(
        geometry=[LineString([(-9, 52), (-7, 52)])],
        crs="EPSG:4326",
    )
    profs = profiles_from_gdf(fake_data, gdf)

    assert profs[0].name == "Feature_1"


def test_from_gdf_out_of_bounds_skipped(fake_data):
    """profiles_from_gdf skips features entirely outside the DEM extent."""
    gdf = geopandas.GeoDataFrame(
        geometry=[
            LineString([(-9, 52), (-7, 52)]),  # inside
            LineString([(10, 10), (20, 10)]),  # outside
        ],
        crs="EPSG:4326",
    )
    profs = profiles_from_gdf(fake_data, gdf)

    assert len(profs) == 1


def test_from_gdf_metadata_stored(fake_data):
    """profiles_from_gdf stores non-geometry columns as profile metadata."""
    gdf = geopandas.GeoDataFrame(
        {"cruise": ["RC1234"]},
        geometry=[LineString([(-9, 52), (-7, 52)])],
        crs="EPSG:4326",
    )
    profs = profiles_from_gdf(fake_data, gdf)

    assert profs[0].metadata["cruise"] == "RC1234"


def test_to_gdf_multiple_profiles(fake_data):
    """to_gdf returns one row per profile with correct CRS."""
    prof1 = extract_profile(
        fake_data, start=(-9, 52), end=(-6, 53), num_points=10, name="P1"
    )
    prof2 = extract_profile(
        fake_data, start=(-9, 53), end=(-6, 54), num_points=10, name="P2"
    )
    gdf = to_gdf([prof1, prof2])

    assert len(gdf) == 2
    assert gdf.crs.to_epsg() == 4326
    assert list(gdf["name"]) == ["P1", "P2"]


def test_to_gdf_empty_raises():
    """to_gdf raises ValueError for empty list."""
    with pytest.raises(ValueError):
        to_gdf([])
