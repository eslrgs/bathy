"""Interactive profile drawing on bathymetry maps."""

from dataclasses import dataclass, field

import cmocean.cm as cmo
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D

from bathy.plot import get_extent
from bathy.profile import Profile, profile_from_coordinates

_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

_PICK_RADIUS_PX = 10


def _point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point *p* to line segment *a*--*b* in 2-D."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0) if denom > 0 else 0.0
    return float(np.linalg.norm(p - (a + t * ab)))


def _style_profile_axis(ax, n_profiles: int) -> None:
    """Apply consistent labels and styling to the profile axis."""
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"{n_profiles} profile{'s' if n_profiles != 1 else ''}")
    ax.grid(True, alpha=0.3)
    if n_profiles:
        ax.legend()


@dataclass
class _ProfileState:
    """Mutable state for one profile on the map."""

    coords: list[tuple[float, float]] = field(default_factory=list)
    line_artist: Line2D | None = None
    marker_artist: Line2D | None = None
    color: str = ""
    finished: bool = False


class _ProfileDrawer:
    """Manage interactive profile drawing and editing.

    Not intended for direct use --- instantiated by ``draw_profile()``.
    """

    def __init__(
        self,
        data: xr.DataArray,
        cmap,
        profiles: list[Profile] | None = None,
    ) -> None:
        self._data = data
        extent = get_extent(data)

        fig, (ax_map, ax_profile) = plt.subplots(1, 2, figsize=(14, 5))
        self._fig = fig
        self._ax_map = ax_map
        self._ax_profile = ax_profile

        ax_map.imshow(
            data.values, cmap=cmap, origin="lower", extent=extent, aspect="auto"
        )
        ax_map.set_xlabel("Longitude (°)")
        ax_map.set_ylabel("Latitude (°)")
        ax_map.set_title("Draw profile (right-click: finish, double-click: done)")

        ax_profile.set_xlabel("Distance (km)")
        ax_profile.set_ylabel("Elevation (m)")
        ax_profile.set_title("Profile")
        ax_profile.grid(True, alpha=0.3)

        self._state: dict = {"profiles": [], "done": False}
        self._profile_states: list[_ProfileState] = []
        self._drag_info: tuple[int, int] | None = None

        if profiles:
            self._load_profiles(profiles)
        self._start_new_profile()

        # Prevent garbage collection: mpl_connect stores bound methods as
        # weak references, so we pin the drawer to the figure.
        fig._profile_drawer = self

        fig.canvas.mpl_connect("button_press_event", self._on_press)
        fig.canvas.mpl_connect("button_release_event", self._on_release)
        fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        fig.canvas.mpl_connect("key_press_event", self._on_key)

    # -- profile management ------------------------------------------------

    @property
    def _active(self) -> _ProfileState:
        """The current in-progress (unfinished) profile."""
        return self._profile_states[-1]

    def _start_new_profile(self) -> None:
        color = _COLORS[len(self._profile_states) % len(_COLORS)]
        (line,) = self._ax_map.plot([], [], "-", color=color, linewidth=2)
        (markers,) = self._ax_map.plot(
            [],
            [],
            "o",
            color=color,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
        self._profile_states.append(
            _ProfileState(
                coords=[], line_artist=line, marker_artist=markers, color=color
            )
        )

    def _load_profiles(self, profiles: list[Profile]) -> None:
        """Populate the map with existing profiles for editing."""
        for prof in profiles:
            lons = prof.metadata.get("path_lons")
            lats = prof.metadata.get("path_lats")
            if lons and lats:
                coords = list(zip(lons, lats))
            else:
                coords = [
                    (prof.start_lon, prof.start_lat),
                    (prof.end_lon, prof.end_lat),
                ]
            self._start_new_profile()
            ps = self._profile_states[-1]
            ps.coords = coords
            ps.finished = True
            self._update_map_artists(len(self._profile_states) - 1)
        self._sync_and_replot()

    def _update_map_artists(self, idx: int) -> None:
        ps = self._profile_states[idx]
        if ps.coords:
            lons, lats = zip(*ps.coords)
        else:
            lons, lats = [], []
        ps.line_artist.set_data(lons, lats)
        ps.marker_artist.set_data(lons, lats)

    def _sync_and_replot(self) -> None:
        """Rebuild ``state["profiles"]`` and redraw the profile axis."""
        self._state["profiles"] = []
        finished = []
        for ps in self._profile_states:
            if ps.finished and len(ps.coords) >= 2:
                name = f"Profile {len(self._state['profiles']) + 1}"
                ps.line_artist.set_label(name)
                self._state["profiles"].append(
                    profile_from_coordinates(self._data, ps.coords, name=name)
                )
                finished.append(ps)

        self._ax_profile.clear()
        for prof, ps in zip(self._state["profiles"], finished):
            self._ax_profile.plot(
                prof.distances / 1000, prof.elevations, color=ps.color, label=prof.name
            )
        _style_profile_axis(self._ax_profile, len(self._state["profiles"]))

    def _redraw(self) -> None:
        self._fig.canvas.draw_idle()

    # -- hit testing -------------------------------------------------------

    def _click_px(self, event) -> np.ndarray:
        """Convert event data coordinates to display pixels."""
        return np.asarray(self._ax_map.transData.transform((event.xdata, event.ydata)))

    def _find_nearest_point(self, event) -> tuple[int, int] | None:
        """Return ``(profile_idx, point_idx)`` of the nearest waypoint, or *None*."""
        click_xy = self._click_px(event)
        best_dist = _PICK_RADIUS_PX
        best: tuple[int, int] | None = None
        for pi, ps in enumerate(self._profile_states):
            for ci, coord in enumerate(ps.coords):
                px = np.asarray(self._ax_map.transData.transform(coord))
                d = float(np.linalg.norm(click_xy - px))
                if d < best_dist:
                    best_dist = d
                    best = (pi, ci)
        return best

    def _find_nearest_segment(self, event) -> tuple[int, int] | None:
        """Nearest line segment as ``(profile_idx, segment_idx)``, or *None*."""
        click_xy = self._click_px(event)
        best_dist = _PICK_RADIUS_PX
        best: tuple[int, int] | None = None
        for pi, ps in enumerate(self._profile_states):
            for si in range(len(ps.coords) - 1):
                a = np.asarray(self._ax_map.transData.transform(ps.coords[si]))
                b = np.asarray(self._ax_map.transData.transform(ps.coords[si + 1]))
                d = _point_to_segment_dist(click_xy, a, b)
                if d < best_dist:
                    best_dist = d
                    best = (pi, si)
        return best

    # -- event handlers ----------------------------------------------------

    def _on_press(self, event) -> None:
        if self._state["done"] or event.inaxes != self._ax_map:
            return

        # Double-click → finish drawing
        if event.dblclick:
            self._finish_drawing()
            return

        # Right-click → finish current profile
        if event.button == MouseButton.RIGHT:
            self._finish_current()
            return

        # Middle-click → undo last point
        if event.button == MouseButton.MIDDLE:
            self._undo_last_point()
            return

        if event.button != MouseButton.LEFT:
            return

        # Shift+click → delete waypoint
        if getattr(event, "key", None) == "shift":
            hit = self._find_nearest_point(event)
            if hit:
                self._delete_point(*hit)
            return

        # Click near point → start drag
        hit = self._find_nearest_point(event)
        if hit:
            self._drag_info = (hit[0], hit[1])
            return

        # Click near segment → insert waypoint
        seg = self._find_nearest_segment(event)
        if seg:
            self._insert_point(seg[0], seg[1], event.xdata, event.ydata)
            return

        # Click on empty space → add waypoint to active profile
        if not self._active.finished:
            self._active.coords.append((event.xdata, event.ydata))
            self._update_map_artists(len(self._profile_states) - 1)
            self._redraw()

    def _on_motion(self, event) -> None:
        if self._drag_info is None:
            return
        if event.inaxes != self._ax_map or event.xdata is None:
            return
        pi, ci = self._drag_info
        self._profile_states[pi].coords[ci] = (event.xdata, event.ydata)
        self._update_map_artists(pi)
        self._redraw()

    def _on_release(self, event) -> None:
        if self._drag_info is None:
            return
        pi, _ = self._drag_info
        ps = self._profile_states[pi]
        self._drag_info = None
        if ps.finished:
            self._sync_and_replot()
            self._redraw()

    def _on_key(self, event) -> None:
        if event.key == "z" and not self._state["done"]:
            self._undo_last_point()

    # -- editing actions ---------------------------------------------------

    def _undo_last_point(self) -> None:
        """Remove the last waypoint from the active profile."""
        ps = self._active
        if ps.finished or not ps.coords:
            return
        ps.coords.pop()
        self._update_map_artists(len(self._profile_states) - 1)
        self._redraw()

    def _delete_point(self, profile_idx: int, point_idx: int) -> None:
        """Remove a waypoint; drop the profile entirely if fewer than 2 remain."""
        ps = self._profile_states[profile_idx]
        ps.coords.pop(point_idx)

        if ps.finished and len(ps.coords) < 2:
            ps.line_artist.remove()
            ps.marker_artist.remove()
            self._profile_states.pop(profile_idx)
            if not self._profile_states or self._profile_states[-1].finished:
                self._start_new_profile()
            self._sync_and_replot()
        elif ps.finished:
            self._update_map_artists(profile_idx)
            self._sync_and_replot()
        else:
            self._update_map_artists(profile_idx)

        self._redraw()

    def _insert_point(
        self, profile_idx: int, segment_idx: int, lon: float, lat: float
    ) -> None:
        """Insert a new waypoint between two existing ones."""
        ps = self._profile_states[profile_idx]
        ps.coords.insert(segment_idx + 1, (lon, lat))
        self._update_map_artists(profile_idx)
        if ps.finished:
            self._sync_and_replot()
        self._redraw()

    def _finish_current(self) -> None:
        """Finish the active profile and prepare for a new one."""
        ps = self._active
        if len(ps.coords) < 2:
            return
        ps.finished = True
        self._sync_and_replot()
        self._start_new_profile()
        self._redraw()

    def _finish_drawing(self) -> None:
        """Finish all drawing and lock the widget."""
        self._drag_info = None
        ps = self._active
        # The single-click that precedes a double-click may have added a
        # duplicate trailing waypoint — drop it before finishing.
        if len(ps.coords) >= 2 and ps.coords[-1] == ps.coords[-2]:
            ps.coords.pop()
        if not ps.finished and len(ps.coords) >= 2:
            ps.finished = True
        # Remove trailing empty/invalid active profile
        if not self._profile_states[-1].finished:
            self._profile_states[-1].line_artist.remove()
            self._profile_states[-1].marker_artist.remove()
            self._profile_states.pop()
        self._sync_and_replot()
        self._state["done"] = True
        n = len(self._state["profiles"])
        self._ax_map.set_title(f"{n} profile{'s' if n != 1 else ''} extracted")
        if n:
            self._ax_map.legend()
        self._redraw()


def draw_profile(
    data: xr.DataArray,
    profiles: list[Profile] | None = None,
    cmap=cmo.deep_r,
) -> dict:
    """
    Interactively draw and edit profiles on a bathymetry map.

    Left-click to add waypoints for the current profile.
    Right-click to finish the current profile and start a new one.
    Double-click to finish drawing entirely.
    Middle-click or press ``z`` to undo the last waypoint.
    Shift-click on a waypoint to delete it from any profile.
    Click on a line segment to insert a new waypoint.
    Drag any waypoint to reposition it; the profile updates on release.

    Pass existing profiles via the *profiles* parameter to reload and
    edit them. This enables a round-trip workflow: draw, save with
    ``to_gdf(...).to_file()``, then reload with ``profiles_from_file``
    and pass back in for further editing.

    Requires the ipympl backend (``%matplotlib widget``) in Jupyter.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data with lon and lat coordinates.
    profiles : list[Profile], optional
        Existing profiles to load onto the map for editing.
    cmap
        Colourmap for bathymetry display.

    Returns
    -------
    dict
        State dictionary with keys:

        - ``"profiles"`` — list of extracted Profile objects
        - ``"done"`` — whether drawing is complete

        For a single profile, access ``result["profiles"][0]``.

    Examples
    --------
    .. code-block:: python

        %matplotlib widget
        import bathy

        data = bathy.load_bathymetry("path/to/data.nc")
        result = bathy.draw_profile(data)
        # Left-click waypoints, right-click to finish each profile
        # Drag waypoints to adjust, shift-click to delete
        # Double-click to stop drawing
        result["profiles"]       # list of all profiles
        result["profiles"][0]    # first profile

        # Save and reload for further editing
        bathy.to_gdf(result["profiles"]).to_file("profiles.gpkg")
        reloaded = bathy.profiles_from_file(data, "profiles.gpkg")
        result = bathy.draw_profile(data, profiles=reloaded)
    """
    drawer = _ProfileDrawer(data, cmap, profiles=profiles)
    return drawer._state
