"""Tests for profile drawing."""

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from bathy.draw import _ProfileDrawingLogic  # noqa: E402
from bathy.profile import extract_profile, profile_from_coordinates  # noqa: E402


@pytest.fixture()
def drawer(fake_data):
    """Create a _ProfileDrawingLogic backed by an Agg figure."""
    fig = Figure(figsize=(14, 5))
    fig.canvas  # noqa: B018 — force canvas creation
    return _ProfileDrawingLogic(fake_data, cmap="viridis", fig=fig)


def _click(drawer, lon, lat, button=MouseButton.LEFT, dblclick=False, key=None):
    """Simulate a mouse click on the map axes."""
    canvas = drawer._fig.canvas
    event = MouseEvent("button_press_event", canvas, 0, 0, button=button, key=key)
    event.inaxes = drawer._ax_map
    event.xdata = lon
    event.ydata = lat
    event.dblclick = dblclick
    canvas.callbacks.process("button_press_event", event)


def _release(drawer, lon, lat, button=MouseButton.LEFT):
    """Simulate a mouse button release on the map axes."""
    canvas = drawer._fig.canvas
    event = MouseEvent("button_release_event", canvas, 0, 0, button=button)
    event.inaxes = drawer._ax_map
    event.xdata = lon
    event.ydata = lat
    canvas.callbacks.process("button_release_event", event)


def _motion(drawer, lon, lat):
    """Simulate mouse motion on the map axes."""
    canvas = drawer._fig.canvas
    event = MouseEvent("motion_notify_event", canvas, 0, 0)
    event.inaxes = drawer._ax_map
    event.xdata = lon
    event.ydata = lat
    canvas.callbacks.process("motion_notify_event", event)


def _key_press(drawer, key):
    """Simulate a key press event."""
    canvas = drawer._fig.canvas
    event = KeyEvent("key_press_event", canvas, 0, 0)
    event.key = key
    canvas.callbacks.process("key_press_event", event)


# -- existing behaviour ---------------------------------------------------


def test_creates_layout(drawer):
    """_ProfileDrawingLogic creates a figure with map and profile axes."""
    assert drawer.profiles == []
    assert drawer.done is False
    assert len(drawer._fig.axes) == 2


def test_right_click_finishes_one_profile(drawer):
    """Right-click extracts the current profile and allows starting another."""
    _click(drawer, -8, 52)
    _click(drawer, -7, 53)
    _click(drawer, -7, 53, button=MouseButton.RIGHT)

    assert len(drawer.profiles) == 1
    assert drawer.profiles[0].name == "Profile 1"
    assert drawer.profiles[0].distances[0] == 0
    assert drawer.done is False


def test_right_click_ignored_with_fewer_than_two_points(drawer):
    """Right-click with fewer than 2 waypoints does nothing."""
    _click(drawer, -8, 52)
    _click(drawer, -8, 52, button=MouseButton.RIGHT)

    assert len(drawer.profiles) == 0
    assert drawer.done is False


def test_multiple_profiles(drawer):
    """Multiple profiles can be drawn sequentially."""
    _click(drawer, -8, 52)
    _click(drawer, -7, 53)
    _click(drawer, -7, 53, button=MouseButton.RIGHT)

    _click(drawer, -9, 51)
    _click(drawer, -6, 52)
    _click(drawer, -6, 52, button=MouseButton.RIGHT)

    assert len(drawer.profiles) == 2
    assert drawer.profiles[0].name == "Profile 1"
    assert drawer.profiles[1].name == "Profile 2"


def test_double_click_finishes(drawer):
    """Double-click finishes drawing entirely."""
    _click(drawer, -8, 52)
    _click(drawer, -7, 53)
    _click(drawer, -7, 53, dblclick=True)

    assert drawer.done is True
    assert len(drawer.profiles) == 1


def test_clicks_ignored_after_done(drawer):
    """Clicks are ignored once drawing is complete."""
    _click(drawer, -8, 52)
    _click(drawer, -7, 53)
    _click(drawer, -7, 53, dblclick=True)

    _click(drawer, -6, 54)

    assert len(drawer.profiles) == 1


# -- undo ------------------------------------------------------------------


def test_undo_middle_click(drawer):
    """Middle-click removes the last waypoint from the active profile."""
    _click(drawer, -8, 52)
    _click(drawer, -7.5, 52.5)
    _click(drawer, -7, 53)
    _click(drawer, -7, 53, button=MouseButton.MIDDLE)
    _click(drawer, -7, 53, button=MouseButton.RIGHT)

    assert len(drawer.profiles) == 1
    assert len(drawer.profiles[0].distances) == 2


def test_undo_z_key(drawer):
    """Pressing 'z' removes the last waypoint from the active profile."""
    _click(drawer, -8, 52)
    _click(drawer, -7.5, 52.5)
    _click(drawer, -7, 53)
    _key_press(drawer, "z")
    _click(drawer, -7, 53, button=MouseButton.RIGHT)

    assert len(drawer.profiles) == 1
    assert len(drawer.profiles[0].distances) == 2


def test_undo_on_empty_does_nothing(drawer):
    """Undo with no waypoints does not crash."""
    _click(drawer, -8, 52, button=MouseButton.MIDDLE)

    assert len(drawer.profiles) == 0
    assert drawer.done is False


def test_undo_does_not_affect_finished_profiles(drawer):
    """Undo only removes points from the active (unfinished) profile."""
    _click(drawer, -8, 52)
    _click(drawer, -7, 53)
    _click(drawer, -7, 53, button=MouseButton.RIGHT)

    _click(drawer, -9, 51)
    _click(drawer, -8, 52, button=MouseButton.MIDDLE)

    assert len(drawer.profiles) == 1
    assert len(drawer.profiles[0].distances) == 2


# -- delete ----------------------------------------------------------------


def test_delete_removes_profile_below_two_points(drawer):
    """Deleting a waypoint that leaves < 2 points removes the finished profile."""
    _click(drawer, -8, 52)
    _click(drawer, -7, 53)
    _click(drawer, -7, 53, button=MouseButton.RIGHT)

    assert len(drawer.profiles) == 1

    _click(drawer, -8, 52, key="shift")

    assert len(drawer.profiles) == 0


def test_double_click_with_no_active_points(drawer):
    """Double-click with no active waypoints finishes without error."""
    _click(drawer, -8, 52)
    _click(drawer, -7, 53)
    _click(drawer, -7, 53, button=MouseButton.RIGHT)
    _click(drawer, -7, 53, dblclick=True)

    assert drawer.done is True
    assert len(drawer.profiles) == 1


def test_double_click_deduplicates_trailing_waypoint(drawer):
    """Double-click drops the duplicate from the preceding single-click."""
    _click(drawer, -8, 52)
    _click(drawer, -7, 53)
    _click(drawer, -7, 53)
    _click(drawer, -7, 53, dblclick=True)

    assert len(drawer.profiles) == 1
    assert len(drawer.profiles[0].distances) == 2


# -- drag ------------------------------------------------------------------


def test_drag_waypoint_on_active_profile(drawer):
    """Dragging a waypoint on the in-progress profile updates its coords."""
    _click(drawer, -8, 52)
    _click(drawer, -7, 53)

    _click(drawer, -8, 52)
    _motion(drawer, -9, 51)
    _release(drawer, -9, 51)

    _click(drawer, -7, 53, button=MouseButton.RIGHT)

    assert len(drawer.profiles) == 1
    prof = drawer.profiles[0]
    assert prof.start_x == -9
    assert prof.start_y == 51


def test_drag_waypoint_on_finished_profile(drawer):
    """Dragging a finished profile's waypoint recalculates the profile."""
    _click(drawer, -8, 52)
    _click(drawer, -7, 53)
    _click(drawer, -7, 53, button=MouseButton.RIGHT)

    original_elevations = drawer.profiles[0].elevations.copy()

    _click(drawer, -8, 52)
    _motion(drawer, -9, 51)
    _release(drawer, -9, 51)

    prof = drawer.profiles[0]
    assert prof.start_x == -9
    assert prof.start_y == 51
    assert not (prof.elevations == original_elevations).all()


def test_drag_does_nothing_without_motion(drawer):
    """Click-and-release on a point without moving leaves the profile unchanged."""
    _click(drawer, -8, 52)
    _click(drawer, -7, 53)
    _click(drawer, -7, 53, button=MouseButton.RIGHT)

    original_distances = drawer.profiles[0].distances.copy()

    _click(drawer, -8, 52)
    _release(drawer, -8, 52)

    assert (drawer.profiles[0].distances == original_distances).all()


# -- insert ----------------------------------------------------------------


def test_insert_on_finished_profile(drawer):
    """Clicking on a segment of a finished profile inserts a new waypoint."""
    _click(drawer, -8, 52)
    _click(drawer, -6, 54)
    _click(drawer, -6, 54, button=MouseButton.RIGHT)

    assert len(drawer.profiles[0].distances) == 2

    _click(drawer, -7, 53)

    assert len(drawer.profiles[0].distances) == 3


def test_insert_on_active_profile(drawer):
    """Clicking on a segment of the in-progress profile inserts a waypoint."""
    _click(drawer, -8, 52)
    _click(drawer, -6, 54)

    _click(drawer, -7, 53)

    _click(drawer, -7, 53, button=MouseButton.RIGHT)

    assert len(drawer.profiles) == 1
    assert len(drawer.profiles[0].distances) == 3


# -- loading existing profiles --------------------------------------------


def test_load_existing_profiles(fake_data):
    """Passing profiles pre-populates the map with editable profiles."""
    p1 = profile_from_coordinates(fake_data, [(-8, 52), (-7, 53)], name="A")
    p2 = profile_from_coordinates(fake_data, [(-9, 51), (-6, 54)], name="B")

    fig = Figure(figsize=(14, 5))
    fig.canvas  # noqa: B018
    d = _ProfileDrawingLogic(fake_data, cmap="viridis", fig=fig, profiles=[p1, p2])

    assert len(d.profiles) == 2
    assert d.profiles[0].name == "Profile 1"
    assert d.profiles[1].name == "Profile 2"


def test_load_profiles_without_path_metadata(fake_data):
    """Profiles from extract_profile fall back to start/end coordinates."""
    prof = extract_profile(fake_data, (-8, 52), (-6, 54), num_points=50)

    fig = Figure(figsize=(14, 5))
    fig.canvas  # noqa: B018
    d = _ProfileDrawingLogic(fake_data, cmap="viridis", fig=fig, profiles=[prof])

    assert len(d.profiles) == 1
    assert len(d.profiles[0].distances) == 2


def test_load_and_draw_more(fake_data):
    """Loaded profiles and newly drawn profiles coexist."""
    existing = profile_from_coordinates(
        fake_data, [(-8, 52), (-7, 53)], name="Existing"
    )
    fig = Figure(figsize=(14, 5))
    fig.canvas  # noqa: B018
    d = _ProfileDrawingLogic(fake_data, cmap="viridis", fig=fig, profiles=[existing])

    _click(d, -9, 51)
    _click(d, -6, 54)
    _click(d, -6, 54, button=MouseButton.RIGHT)

    assert len(d.profiles) == 2
    assert d.profiles[0].name == "Profile 1"
    assert d.profiles[1].name == "Profile 2"
