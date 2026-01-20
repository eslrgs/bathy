"""Tests for profile module."""

import numpy as np

from bathy.profile import Profile


def test_create_profile(fake_data):
    """Create a profile between two points."""
    prof = Profile(fake_data, -9, 52, -6, 53, num_points=10, name="Test")

    assert prof.name == "Test"
    assert prof.num_points == 10
    assert len(prof.distances) == 10
    assert len(prof.elevations) == 10
    assert prof.start_lon == -9
    assert prof.start_lat == 52
    assert prof.end_lon == -6
    assert prof.end_lat == 53


def test_max_depth(fake_data):
    """Find the deepest point in a profile."""
    prof = Profile(fake_data, -9, 52, -6, 53, num_points=10)
    distance, depth = prof.max_depth()

    assert distance >= 0
    assert depth < 0


def test_gradient(fake_data):
    """Calculate profile gradient."""
    prof = Profile(fake_data, -9, 52, -6, 53, num_points=10)
    gradient = prof.gradient()

    assert len(gradient) == 10
    assert isinstance(gradient, np.ndarray)


def test_distance_axis_starts_at_start_point(fake_data):
    """Verify distance axis starts at zero for user-defined start point."""
    prof = Profile(fake_data, -9, 52, -6, 53, num_points=10)

    # Distance should start at 0
    assert prof.distances[0] == 0
    # Distances should be monotonically increasing
    assert np.all(np.diff(prof.distances) > 0)


def test_ensure_descending_bathymetry():
    """Test _ensure_descending with bathymetric (negative) elevations."""
    distances = np.array([0, 10, 20, 30])

    # Case 1: Already descending (shallow to deep: -100 to -4000)
    elevations_desc = np.array([-100, -1000, -2000, -4000])
    dist_out, elev_out = Profile._ensure_descending(distances, elevations_desc)
    assert np.array_equal(dist_out, distances)
    assert np.array_equal(elev_out, elevations_desc)

    # Case 2: Ascending (deep to shallow: -4000 to -100), should flip
    elevations_asc = np.array([-4000, -2000, -1000, -100])
    dist_out, elev_out = Profile._ensure_descending(distances, elevations_asc)
    assert np.array_equal(dist_out, distances[::-1])
    assert np.array_equal(elev_out, elevations_asc[::-1])


def test_ensure_descending_topography():
    """Test _ensure_descending with topographic (positive) elevations."""
    distances = np.array([0, 10, 20, 30])

    # Case 1: Already descending (high to low: 1000 to 100)
    elevations_desc = np.array([1000, 500, 300, 100])
    dist_out, elev_out = Profile._ensure_descending(distances, elevations_desc)
    assert np.array_equal(dist_out, distances)
    assert np.array_equal(elev_out, elevations_desc)

    # Case 2: Ascending (low to high: 100 to 1000), should flip
    elevations_asc = np.array([100, 300, 500, 1000])
    dist_out, elev_out = Profile._ensure_descending(distances, elevations_asc)
    assert np.array_equal(dist_out, distances[::-1])
    assert np.array_equal(elev_out, elevations_asc[::-1])
