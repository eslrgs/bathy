"""Smoke tests for plotting functions."""

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import bathy  # noqa: E402
from bathy.plot import get_extent  # noqa: E402
from bathy.profile import extract_profile, knickpoints  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all figures after each test to free memory."""
    yield
    plt.close("all")


# -- Grid plot functions --


def test_plot_bathy(fake_data):
    fig, ax = bathy.plot_bathy(fake_data)
    assert isinstance(fig, Figure)


def test_plot_bathy_no_mask(fake_data):
    fig, ax = bathy.plot_bathy(fake_data, mask_land=False)
    assert isinstance(fig, Figure)


def test_plot_hillshade(fake_data):
    fig, ax = bathy.plot_hillshade(fake_data)
    assert isinstance(fig, Figure)


def test_plot_slope(fake_data):
    fig, ax = bathy.plot_slope(fake_data)
    assert isinstance(fig, Figure)


def test_plot_curvature(fake_data):
    fig, ax = bathy.plot_curvature(fake_data)
    assert isinstance(fig, Figure)


def test_plot_bpi(fake_data):
    fig, ax = bathy.plot_bpi(fake_data)
    assert isinstance(fig, Figure)


def test_plot_rugosity(fake_data):
    fig, ax = bathy.plot_rugosity(fake_data)
    assert isinstance(fig, Figure)


def test_plot_aspect(fake_data):
    fig, ax = bathy.plot_aspect(fake_data)
    assert isinstance(fig, Figure)


def test_plot_geomorphons(fake_data):
    fig, ax = bathy.plot_geomorphons(fake_data)
    assert isinstance(fig, Figure)


def test_plot_overview(fake_data):
    fig, axes = bathy.plot_overview(fake_data)
    assert isinstance(fig, Figure)
    assert axes.shape == (4, 2)


def test_plot_histogram(fake_data):
    fig, ax = bathy.plot_histogram(fake_data)
    assert isinstance(fig, Figure)


def test_plot_depth_zones(fake_data):
    fig, ax = bathy.plot_depth_zones(fake_data)
    assert isinstance(fig, Figure)


def test_plot_surface3d(fake_data):
    fig, ax = bathy.plot_surface3d(fake_data, stride=2)
    assert isinstance(fig, Figure)


def test_plot_hypsometric_curve(fake_data):
    fig, ax = bathy.plot_hypsometric_curve(fake_data)
    assert isinstance(fig, Figure)


# -- Profile plot functions --


def test_plot_profile(fake_profile):
    fig, axes = bathy.plot_profile(fake_profile)
    assert isinstance(fig, Figure)
    assert len(axes) == 1


def test_plot_profile_with_map(fake_data, fake_profile):
    fig, axes = bathy.plot_profile(
        fake_profile, show_map=True, bathymetry_data=fake_data
    )
    assert isinstance(fig, Figure)
    assert len(axes) == 2


def test_plot_profiles(fake_data):
    prof1 = extract_profile(
        fake_data, start=(-9, 52), end=(-6, 53), num_points=10, name="A"
    )
    prof2 = extract_profile(
        fake_data, start=(-9, 53), end=(-6, 54), num_points=10, name="B"
    )
    fig, axes = bathy.plot_profiles([prof1, prof2])
    assert isinstance(fig, Figure)


def test_plot_profiles_grid(fake_data):
    profs = [
        extract_profile(
            fake_data, start=(-9, 52), end=(-6, 53), num_points=10, name=f"P{i}"
        )
        for i in range(4)
    ]
    fig, axes = bathy.plot_profiles_grid(profs, cols=2)
    assert isinstance(fig, Figure)


def test_plot_profiles_map(fake_data, fake_profile):
    fig, ax = bathy.plot_profiles_map(fake_profile, bathymetry_data=fake_data)
    assert isinstance(fig, Figure)


def test_plot_gradient(fake_profile):
    fig, axes = bathy.plot_gradient(fake_profile)
    assert isinstance(fig, Figure)
    assert len(axes) == 1


def test_plot_knickpoints(fake_data):
    prof = extract_profile(fake_data, start=(-9, 52), end=(-6, 53), num_points=50)
    kp = knickpoints(prof)
    fig, axes = bathy.plot_knickpoints(prof, kp)
    assert isinstance(fig, Figure)


def test_plot_canyons(fake_data):
    prof = extract_profile(fake_data, start=(-9, 52), end=(-6, 53), num_points=50)
    fig, axes = bathy.plot_canyons(prof)
    assert isinstance(fig, Figure)


# -- Helpers --


def test_get_extent():
    """get_extent returns [lon_min, lon_max, lat_min, lat_max]."""
    data = xr.DataArray(
        np.zeros((3, 3)),
        coords={"lon": [0.0, 1.0, 2.0], "lat": [50.0, 51.0, 52.0]},
        dims=["lat", "lon"],
    )
    assert get_extent(data) == [0.0, 2.0, 50.0, 52.0]
