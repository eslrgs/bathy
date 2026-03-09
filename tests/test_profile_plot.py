"""Smoke tests for profile plotting functions."""

from matplotlib.figure import Figure

import bathy
from bathy.profile import extract_profile, knickpoints


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
    canyons = bathy.get_canyons(prof, prominence=5)
    fig, axes = bathy.plot_canyons(prof, canyons)
    assert isinstance(fig, Figure)
