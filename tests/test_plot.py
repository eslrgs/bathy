"""Smoke tests for grid plotting functions."""

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

import bathy
from bathy.plot import get_extent


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


# -- Helpers --


def test_get_extent():
    """get_extent returns [lon_min, lon_max, lat_min, lat_max]."""
    data = xr.DataArray(
        np.zeros((3, 3)),
        coords={"lon": [0.0, 1.0, 2.0], "lat": [50.0, 51.0, 52.0]},
        dims=["lat", "lon"],
    )
    assert get_extent(data) == [0.0, 2.0, 50.0, 52.0]
