"""Tests for interactive profile drawing."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.backend_bases import MouseButton, MouseEvent  # noqa: E402

from bathy.interactive import draw_profile  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all figures after each test."""
    yield
    plt.close("all")


def _click(fig, ax, lon, lat, button=MouseButton.LEFT, dblclick=False):
    """Simulate a mouse click on the given axes."""
    event = MouseEvent("button_press_event", fig.canvas, 0, 0, button=button)
    event.inaxes = ax
    event.xdata = lon
    event.ydata = lat
    event.dblclick = dblclick
    fig.canvas.callbacks.process("button_press_event", event)


def test_creates_layout(fake_data):
    """draw_profile creates a figure with map and profile axes."""
    state = draw_profile(fake_data)

    assert state["profiles"] == []
    assert state["done"] is False

    fig = plt.gcf()
    assert len(fig.axes) == 2


def test_right_click_finishes_one_profile(fake_data):
    """Right-click extracts the current profile and allows starting another."""
    state = draw_profile(fake_data)
    fig = plt.gcf()
    ax_map = fig.axes[0]

    _click(fig, ax_map, -8, 52)
    _click(fig, ax_map, -7, 53)
    _click(fig, ax_map, -7, 53, button=MouseButton.RIGHT)

    assert len(state["profiles"]) == 1
    assert state["profiles"][0].name == "Profile 1"
    assert state["profiles"][0].distances[0] == 0
    assert state["done"] is False


def test_right_click_ignored_with_fewer_than_two_points(fake_data):
    """Right-click with fewer than 2 waypoints does nothing."""
    state = draw_profile(fake_data)
    fig = plt.gcf()
    ax_map = fig.axes[0]

    _click(fig, ax_map, -8, 52)
    _click(fig, ax_map, -8, 52, button=MouseButton.RIGHT)

    assert len(state["profiles"]) == 0
    assert state["done"] is False


def test_multiple_profiles(fake_data):
    """Multiple profiles can be drawn sequentially."""
    state = draw_profile(fake_data)
    fig = plt.gcf()
    ax_map = fig.axes[0]

    _click(fig, ax_map, -8, 52)
    _click(fig, ax_map, -7, 53)
    _click(fig, ax_map, -7, 53, button=MouseButton.RIGHT)

    _click(fig, ax_map, -9, 51)
    _click(fig, ax_map, -6, 52)
    _click(fig, ax_map, -6, 52, button=MouseButton.RIGHT)

    assert len(state["profiles"]) == 2
    assert state["profiles"][0].name == "Profile 1"
    assert state["profiles"][1].name == "Profile 2"


def test_double_click_finishes(fake_data):
    """Double-click finishes drawing entirely."""
    state = draw_profile(fake_data)
    fig = plt.gcf()
    ax_map = fig.axes[0]

    _click(fig, ax_map, -8, 52)
    _click(fig, ax_map, -7, 53)
    _click(fig, ax_map, -7, 53, dblclick=True)

    assert state["done"] is True
    assert len(state["profiles"]) == 1


def test_clicks_ignored_after_done(fake_data):
    """Clicks are ignored once drawing is complete."""
    state = draw_profile(fake_data)
    fig = plt.gcf()
    ax_map = fig.axes[0]

    _click(fig, ax_map, -8, 52)
    _click(fig, ax_map, -7, 53)
    _click(fig, ax_map, -7, 53, dblclick=True)

    _click(fig, ax_map, -6, 54)

    assert len(state["profiles"]) == 1
