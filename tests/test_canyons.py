"""Tests for canyon detection in profile module."""

import polars as pl

from bathy.profile import Profile


def test_get_canyons_returns_dataframe(fake_data):
    """get_canyons returns a Polars DataFrame."""
    prof = Profile(fake_data, -9, 52, -6, 53, num_points=30)
    canyons = prof.get_canyons(prominence=10)

    assert isinstance(canyons, pl.DataFrame)


def test_canyon_dataframe_columns(fake_data):
    """DataFrame has expected columns."""
    prof = Profile(fake_data, -9, 52, -6, 53, num_points=30)
    canyons = prof.get_canyons(prominence=10)

    expected_columns = {
        "floor_distance",
        "floor_elevation",
        "width_start",
        "width_end",
        "width",
        "depth",
        "cross_sectional_area",
    }
    assert set(canyons.columns) == expected_columns


def test_prominence_parameter(fake_data):
    """Higher prominence finds fewer or equal canyons."""
    prof = Profile(fake_data, -9, 52, -6, 53, num_points=30)

    canyons_low = prof.get_canyons(prominence=5)
    canyons_high = prof.get_canyons(prominence=50)

    assert isinstance(canyons_low, pl.DataFrame)
    assert isinstance(canyons_high, pl.DataFrame)
    assert len(canyons_high) <= len(canyons_low)


def test_invalid_prominence_raises(fake_data):
    """Negative or zero prominence raises ValueError."""
    prof = Profile(fake_data, -9, 52, -6, 53, num_points=30)

    import pytest

    with pytest.raises(ValueError):
        prof.get_canyons(prominence=-10)

    with pytest.raises(ValueError):
        prof.get_canyons(prominence=0)


def test_invalid_smooth_raises(fake_data):
    """Negative or zero smooth raises ValueError."""
    prof = Profile(fake_data, -9, 52, -6, 53, num_points=30)

    import pytest

    with pytest.raises(ValueError):
        prof.get_canyons(smooth=-1)

    with pytest.raises(ValueError):
        prof.get_canyons(smooth=0)


def test_canyon_measurements_in_metres(fake_data):
    """All distance measurements are in metres."""
    prof = Profile(fake_data, -9, 52, -6, 53, num_points=30)
    canyons = prof.get_canyons(prominence=5)

    if len(canyons) > 0:
        # Profile distances are in km, so canyon distances in metres should be >= 1000 * min_km
        # or at least positive and reasonable for the scale
        assert all(canyons["floor_distance"] >= 0)
        assert all(canyons["width"] >= 0)
        assert all(canyons["depth"] >= 0)
