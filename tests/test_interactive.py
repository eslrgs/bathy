"""Tests for interactive profile drawing."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent  # noqa: E402

from bathy.interactive import draw_profile  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all figures after each test."""
    yield
    plt.close("all")


def _click(fig, ax, lon, lat, button=MouseButton.LEFT, dblclick=False, key=None):
    """Simulate a mouse click on the given axes."""
    event = MouseEvent("button_press_event", fig.canvas, 0, 0, button=button, key=key)
    event.inaxes = ax
    event.xdata = lon
    event.ydata = lat
    event.dblclick = dblclick
    fig.canvas.callbacks.process("button_press_event", event)


def _release(fig, ax, lon, lat, button=MouseButton.LEFT):
    """Simulate a mouse button release on the given axes."""
    event = MouseEvent("button_release_event", fig.canvas, 0, 0, button=button)
    event.inaxes = ax
    event.xdata = lon
    event.ydata = lat
    fig.canvas.callbacks.process("button_release_event", event)


def _motion(fig, ax, lon, lat):
    """Simulate mouse motion on the given axes."""
    event = MouseEvent("motion_notify_event", fig.canvas, 0, 0)
    event.inaxes = ax
    event.xdata = lon
    event.ydata = lat
    fig.canvas.callbacks.process("motion_notify_event", event)


def _key_press(fig, key):
    """Simulate a key press event."""
    event = KeyEvent("key_press_event", fig.canvas, 0, 0)
    event.key = key
    fig.canvas.callbacks.process("key_press_event", event)


# -- existing behaviour ---------------------------------------------------


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


# -- undo ------------------------------------------------------------------


def test_undo_middle_click(fake_data):
    """Middle-click removes the last waypoint from the active profile."""
    state = draw_profile(fake_data)
    fig = plt.gcf()
    ax_map = fig.axes[0]

    _click(fig, ax_map, -8, 52)
    _click(fig, ax_map, -7.5, 52.5)
    _click(fig, ax_map, -7, 53)
    _click(fig, ax_map, -7, 53, button=MouseButton.MIDDLE)
    _click(fig, ax_map, -7, 53, button=MouseButton.RIGHT)

    assert len(state["profiles"]) == 1
    assert len(state["profiles"][0].distances) == 2


def test_undo_z_key(fake_data):
    """Pressing 'z' removes the last waypoint from the active profile."""
    state = draw_profile(fake_data)
    fig = plt.gcf()
    ax_map = fig.axes[0]

    _click(fig, ax_map, -8, 52)
    _click(fig, ax_map, -7.5, 52.5)
    _click(fig, ax_map, -7, 53)
    _key_press(fig, "z")
    _click(fig, ax_map, -7, 53, button=MouseButton.RIGHT)

    assert len(state["profiles"]) == 1
    assert len(state["profiles"][0].distances) == 2


def test_undo_on_empty_does_nothing(fake_data):
    """Undo with no waypoints does not crash."""
    state = draw_profile(fake_data)
    fig = plt.gcf()
    ax_map = fig.axes[0]

    _click(fig, ax_map, -8, 52, button=MouseButton.MIDDLE)

    assert len(state["profiles"]) == 0
    assert state["done"] is False


def test_undo_does_not_affect_finished_profiles(fake_data):
    """Undo only removes points from the active (unfinished) profile."""
    state = draw_profile(fake_data)
    fig = plt.gcf()
    ax_map = fig.axes[0]

    # Finish a profile
    _click(fig, ax_map, -8, 52)
    _click(fig, ax_map, -7, 53)
    _click(fig, ax_map, -7, 53, button=MouseButton.RIGHT)

    # Start a new one and add one point
    _click(fig, ax_map, -9, 51)
    _click(fig, ax_map, -8, 52, button=MouseButton.MIDDLE)

    # The finished profile is untouched
    assert len(state["profiles"]) == 1
    assert len(state["profiles"][0].distances) == 2


# -- delete ----------------------------------------------------------------


def test_delete_removes_profile_below_two_points(fake_data):
    """Deleting a waypoint that leaves < 2 points removes the finished profile."""
    state = draw_profile(fake_data)
    fig = plt.gcf()
    ax_map = fig.axes[0]

    _click(fig, ax_map, -8, 52)
    _click(fig, ax_map, -7, 53)
    _click(fig, ax_map, -7, 53, button=MouseButton.RIGHT)

    assert len(state["profiles"]) == 1

    # Shift+click on the first waypoint to delete it (only 1 remains → remove)
    _click(fig, ax_map, -8, 52, key="shift")

    assert len(state["profiles"]) == 0


def test_double_click_with_no_active_points(fake_data):
    """Double-click with no active waypoints finishes without error."""
    state = draw_profile(fake_data)
    fig = plt.gcf()
    ax_map = fig.axes[0]

    _click(fig, ax_map, -8, 52)
    _click(fig, ax_map, -7, 53)
    _click(fig, ax_map, -7, 53, button=MouseButton.RIGHT)
    _click(fig, ax_map, -7, 53, dblclick=True)

    assert state["done"] is True
    assert len(state["profiles"]) == 1


def test_double_click_deduplicates_trailing_waypoint(fake_data):
    """Double-click drops the duplicate from the preceding single-click."""
    state = draw_profile(fake_data)
    fig = plt.gcf()
    ax_map = fig.axes[0]

    _click(fig, ax_map, -8, 52)
    _click(fig, ax_map, -7, 53)
    # Simulate the single-click that matplotlib fires before the double-click
    _click(fig, ax_map, -7, 53)
    _click(fig, ax_map, -7, 53, dblclick=True)

    assert len(state["profiles"]) == 1
    assert len(state["profiles"][0].distances) == 2


# -- drag ------------------------------------------------------------------


def test_drag_waypoint_on_active_profile(fake_data):
    """Dragging a waypoint on the in-progress profile updates its coords."""
    state = draw_profile(fake_data)
    fig = plt.gcf()
    ax_map = fig.axes[0]

    _click(fig, ax_map, -8, 52)
    _click(fig, ax_map, -7, 53)

    # Drag the first waypoint from (-8, 52) to (-9, 51)
    _click(fig, ax_map, -8, 52)  # starts drag (hits existing point)
    _motion(fig, ax_map, -9, 51)
    _release(fig, ax_map, -9, 51)

    # Finish and verify the profile uses the new position
    _click(fig, ax_map, -7, 53, button=MouseButton.RIGHT)

    assert len(state["profiles"]) == 1
    prof = state["profiles"][0]
    assert prof.start_lon == -9
    assert prof.start_lat == 51


def test_drag_waypoint_on_finished_profile(fake_data):
    """Dragging a finished profile's waypoint recalculates the profile."""
    state = draw_profile(fake_data)
    fig = plt.gcf()
    ax_map = fig.axes[0]

    _click(fig, ax_map, -8, 52)
    _click(fig, ax_map, -7, 53)
    _click(fig, ax_map, -7, 53, button=MouseButton.RIGHT)

    original_elevations = state["profiles"][0].elevations.copy()

    # Drag the first waypoint to a new location
    _click(fig, ax_map, -8, 52)  # starts drag
    _motion(fig, ax_map, -9, 51)
    _release(fig, ax_map, -9, 51)

    # Profile should have been recalculated with new start
    prof = state["profiles"][0]
    assert prof.start_lon == -9
    assert prof.start_lat == 51
    assert not (prof.elevations == original_elevations).all()


def test_drag_does_nothing_without_motion(fake_data):
    """Click-and-release on a point without moving leaves the profile unchanged."""
    state = draw_profile(fake_data)
    fig = plt.gcf()
    ax_map = fig.axes[0]

    _click(fig, ax_map, -8, 52)
    _click(fig, ax_map, -7, 53)
    _click(fig, ax_map, -7, 53, button=MouseButton.RIGHT)

    original_distances = state["profiles"][0].distances.copy()

    # Click on the waypoint and release without moving
    _click(fig, ax_map, -8, 52)
    _release(fig, ax_map, -8, 52)

    assert (state["profiles"][0].distances == original_distances).all()


# -- insert ----------------------------------------------------------------


def test_insert_on_finished_profile(fake_data):
    """Clicking on a segment of a finished profile inserts a new waypoint."""
    state = draw_profile(fake_data)
    fig = plt.gcf()
    ax_map = fig.axes[0]

    _click(fig, ax_map, -8, 52)
    _click(fig, ax_map, -6, 54)
    _click(fig, ax_map, -6, 54, button=MouseButton.RIGHT)

    assert len(state["profiles"][0].distances) == 2

    # Click on the midpoint of the segment
    _click(fig, ax_map, -7, 53)

    assert len(state["profiles"][0].distances) == 3


def test_insert_on_active_profile(fake_data):
    """Clicking on a segment of the in-progress profile inserts a waypoint."""
    state = draw_profile(fake_data)
    fig = plt.gcf()
    ax_map = fig.axes[0]

    _click(fig, ax_map, -8, 52)
    _click(fig, ax_map, -6, 54)

    # Click on the midpoint of the segment
    _click(fig, ax_map, -7, 53)

    # Finish and verify 3 waypoints
    _click(fig, ax_map, -7, 53, button=MouseButton.RIGHT)

    assert len(state["profiles"]) == 1
    assert len(state["profiles"][0].distances) == 3
