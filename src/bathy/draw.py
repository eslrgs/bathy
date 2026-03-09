"""Interactive profile drawing on bathymetry maps (PyQt6 desktop window)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import cmocean.cm as cmo
import numpy as np
import xarray as xr
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from bathy.plot import get_extent
from bathy.profile import Profile, profile_from_coordinates, profiles_from_file, to_gdf
from bathy.utils import axis_labels

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
    visible: bool = True


class _ProfileDrawingLogic:
    """Core profile drawing logic on a matplotlib Figure.

    No Qt dependency --- the Figure and canvas can be backed by any
    matplotlib backend.  Instantiated by ``draw_profile()`` (which wraps
    it in a PyQt6 window) or directly in tests.
    """

    def __init__(
        self,
        data: xr.DataArray,
        cmap,
        fig: Figure,
        profiles: list[Profile] | None = None,
    ) -> None:
        self._data = data
        self._fig = fig
        self.profiles: list[Profile] = []
        self._finished_states: list[_ProfileState] = []
        self._profile_states: list[_ProfileState] = []
        self._drag_info: tuple[int, int] | None = None
        self.done = False
        self.on_cursor_move = None
        self.on_profiles_changed = None
        self.on_finish_requested = None  # called instead of finishing directly

        self._ax_map, self._ax_profile = fig.subplots(1, 2)

        extent = get_extent(data)
        self._ax_map.imshow(
            data.values, cmap=cmap, origin="lower", extent=extent, aspect="auto"
        )
        x_label, y_label = axis_labels(data)
        self._ax_map.set_xlabel(x_label)
        self._ax_map.set_ylabel(y_label)
        self._ax_map.set_title("Profile drawing")

        _style_profile_axis(self._ax_profile, 0)

        if profiles:
            self._load_profiles(profiles)
        self._start_new_profile()

        fig.canvas.mpl_connect("button_press_event", self._on_press)
        fig.canvas.mpl_connect("button_release_event", self._on_release)
        fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        fig.canvas.mpl_connect("key_press_event", self._on_key)

        # prevent gc: mpl_connect stores weak references
        fig._profile_drawing = self  # type: ignore[attr-defined]

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
            xs = prof.metadata.get("path_xs")
            ys = prof.metadata.get("path_ys")
            if xs and ys:
                coords = list(zip(xs, ys))
            else:
                coords = [
                    (prof.start_x, prof.start_y),
                    (prof.end_x, prof.end_y),
                ]
            self._start_new_profile()
            ps = self._active
            ps.coords = coords
            ps.finished = True
            self._update_map_artists(len(self._profile_states) - 1)
        self._sync_and_replot()

    def _update_map_artists(self, idx: int) -> None:
        ps = self._profile_states[idx]
        if ps.coords:
            xs, ys = zip(*ps.coords)
        else:
            xs, ys = [], []
        ps.line_artist.set_data(xs, ys)
        ps.marker_artist.set_data(xs, ys)

    def _sync_and_replot(self) -> None:
        """Rebuild ``self.profiles`` and redraw the profile axis."""
        self.profiles = []
        self._finished_states = []
        for ps in self._profile_states:
            if ps.finished and len(ps.coords) >= 2:
                name = f"Profile {len(self.profiles) + 1}"
                ps.line_artist.set_label(name)
                self.profiles.append(
                    profile_from_coordinates(self._data, ps.coords, name=name)
                )
                self._finished_states.append(ps)

        self._replot_profile_axis()

        if self.on_profiles_changed:
            self.on_profiles_changed()

    def _replot_profile_axis(self) -> None:
        """Redraw the profile axis, respecting visibility flags."""
        self._ax_profile.clear()
        visible_count = 0
        for prof, ps in zip(self.profiles, self._finished_states):
            if ps.visible:
                self._ax_profile.plot(
                    prof.distances / 1000,
                    prof.elevations,
                    color=ps.color,
                    label=prof.name,
                )
                visible_count += 1
        _style_profile_axis(self._ax_profile, visible_count)

    def set_visibility(self, profile_idx: int, visible: bool) -> None:
        """Show or hide a finished profile on the map and profile plot."""
        if profile_idx >= len(self._finished_states):
            return
        ps = self._finished_states[profile_idx]
        ps.visible = visible
        ps.line_artist.set_visible(visible)
        ps.marker_artist.set_visible(visible)
        self._replot_profile_axis()
        self._redraw()

    def set_all_visible(self, visible: bool) -> None:
        """Show or hide all finished profiles at once."""
        for ps in self._finished_states:
            ps.visible = visible
            ps.line_artist.set_visible(visible)
            ps.marker_artist.set_visible(visible)
        self._replot_profile_axis()
        self._redraw()

    def _redraw(self) -> None:
        self._fig.canvas.draw_idle()

    # -- hit testing -------------------------------------------------------

    def _click_px(self, event) -> np.ndarray:
        """Convert event data coordinates to display pixels."""
        return np.asarray(self._ax_map.transData.transform((event.xdata, event.ydata)))

    def _find_nearest_point(self, event) -> tuple[int, int] | None:
        """Return ``(profile_idx, point_idx)`` of the nearest waypoint."""
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
        """Nearest line segment as ``(profile_idx, segment_idx)``."""
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
        if self.done or event.inaxes != self._ax_map:
            return

        if event.dblclick:
            self._finish_drawing()
            return

        if event.button == MouseButton.RIGHT:
            self._finish_current()
            return

        if event.button == MouseButton.MIDDLE:
            self._undo_last_point()
            return

        if event.button != MouseButton.LEFT:
            return

        if getattr(event, "key", None) == "shift":
            hit = self._find_nearest_point(event)
            if hit:
                self._delete_point(*hit)
            return

        hit = self._find_nearest_point(event)
        if hit:
            self._drag_info = hit
            return

        seg = self._find_nearest_segment(event)
        if seg:
            self._insert_point(seg[0], seg[1], event.xdata, event.ydata)
            return

        if not self._active.finished:
            self._active.coords.append((event.xdata, event.ydata))
            self._update_map_artists(len(self._profile_states) - 1)
            self._redraw()

    def _on_motion(self, event) -> None:
        if event.inaxes == self._ax_map and event.xdata is not None:
            if self.on_cursor_move:
                self.on_cursor_move(event.xdata, event.ydata)
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
        if event.key == "z" and not self.done:
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
        else:
            self._update_map_artists(profile_idx)

        if ps.finished:
            self._sync_and_replot()
        self._redraw()

    def _insert_point(
        self, profile_idx: int, segment_idx: int, x: float, y: float
    ) -> None:
        """Insert a new waypoint between two existing ones."""
        ps = self._profile_states[profile_idx]
        ps.coords.insert(segment_idx + 1, (x, y))
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
        """Finish all drawing (or delegate to on_finish_requested)."""
        if self.on_finish_requested:
            self.on_finish_requested()
            return
        self._do_finish()

    def _do_finish(self) -> None:
        """Unconditionally finish all drawing."""
        self._drag_info = None
        ps = self._active
        # The single-click that precedes a double-click may have added a
        # duplicate trailing waypoint — drop it before finishing.
        if len(ps.coords) >= 2 and ps.coords[-1] == ps.coords[-2]:
            ps.coords.pop()
        if not ps.finished and len(ps.coords) >= 2:
            ps.finished = True
        # Remove trailing empty/invalid active profile
        if not ps.finished:
            ps.line_artist.remove()
            ps.marker_artist.remove()
            self._profile_states.pop()
        self._sync_and_replot()
        self.done = True
        n = len(self.profiles)
        self._ax_map.set_title(f"{n} profile{'s' if n != 1 else ''} extracted")
        if n:
            self._ax_map.legend()
        self._redraw()


def draw_profile(
    data: xr.DataArray,
    profiles: list[Profile] | None = None,
    cmap=cmo.deep_r,
) -> list[Profile]:
    """
    Interactively draw and edit profiles on a bathymetry map.

    Opens a desktop window with a bathymetry map and a live profile plot.
    Left-click to add waypoints for the current profile.
    Right-click to finish the current profile and start a new one.
    Double-click or press the Done button to finish drawing entirely.
    Middle-click or press ``z`` to undo the last waypoint.
    Shift-click on a waypoint to delete it from any profile.
    Click on a line segment to insert a new waypoint.
    Drag any waypoint to reposition it; the profile updates on release.

    Pass existing profiles via the *profiles* parameter to reload and
    edit them. This enables a round-trip workflow: draw, save with
    ``to_gdf(...).to_file()``, then reload with ``profiles_from_file``
    and pass back in for further editing.

    Requires PyQt6: ``uv pip install bathy[draw]``.

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
    list[Profile]
        Extracted profiles. Empty list if no valid profiles were drawn.

    Examples
    --------
    .. code-block:: python

        import bathy

        data = bathy.load_bathymetry("path/to/data.nc")
        profiles = bathy.draw_profile(data)
        # Left-click waypoints, right-click to finish each profile
        # Drag waypoints to adjust, shift-click to delete
        # Double-click or press Done to stop drawing
        profiles[0]  # first profile

        # Save and reload for further editing
        bathy.to_gdf(profiles).to_file("profiles.gpkg")
        reloaded = bathy.profiles_from_file(data, "profiles.gpkg")
        profiles = bathy.draw_profile(data, profiles=reloaded)
    """
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qtagg import (
            NavigationToolbar2QT as NavigationToolbar,
        )
        from PyQt6.QtCore import QEventLoop, Qt
        from PyQt6.QtGui import QFont, QIcon, QPainter, QPixmap
        from PyQt6.QtWidgets import (
            QApplication,
            QFileDialog,
            QHBoxLayout,
            QLabel,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QSplitter,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise ImportError(
            "PyQt6 is required for draw_profile(). "
            "Install it with: uv pip install bathy[draw]"
        ) from exc

    class _ProfileWindow(QMainWindow):
        """Qt window wrapping the profile drawing logic."""

        def __init__(self, logic: _ProfileDrawingLogic) -> None:
            super().__init__()
            self.setWindowTitle("bathy — Draw Profile")
            self.setMinimumSize(1100, 450)

            # App icon from emoji
            pixmap = QPixmap(256, 256)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setFont(QFont("Apple Color Emoji", 200))
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "\U0001f310")
            painter.end()
            icon = QIcon(pixmap)
            self.setWindowIcon(icon)
            QApplication.instance().setWindowIcon(icon)
            self._logic = logic
            self._updating_list = False

            canvas = FigureCanvasQTAgg(logic._fig)
            canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
            canvas.mpl_connect("button_press_event", lambda _: canvas.setFocus())
            self._canvas = canvas

            # -- matplotlib navigation toolbar (pan / zoom / home) --
            toolbar = NavigationToolbar(canvas, self)

            # -- coordinate readout --
            self._coord_label = QLabel("")
            self._coord_label.setFixedWidth(220)
            self._coord_label.setStyleSheet(
                "font-family: monospace; color: #333; padding: 2px 6px;"
            )
            logic.on_cursor_move = self._update_coords

            # -- profile list (checkboxes to toggle visibility) --
            self._profile_list = QListWidget()
            self._profile_list.itemChanged.connect(self._on_item_toggled)
            logic.on_profiles_changed = self._refresh_profile_list

            all_btn = QPushButton("All")
            all_btn.clicked.connect(lambda: self._set_all_visible(True))
            none_btn = QPushButton("None")
            none_btn.clicked.connect(lambda: self._set_all_visible(False))

            list_buttons = QHBoxLayout()
            list_buttons.addWidget(all_btn)
            list_buttons.addWidget(none_btn)

            list_panel = QVBoxLayout()
            list_panel.addLayout(list_buttons)
            list_panel.addWidget(self._profile_list)
            list_widget = QWidget()
            list_widget.setLayout(list_panel)
            list_widget.setMaximumWidth(180)
            list_widget.setMinimumWidth(120)

            # -- hint bar --
            hints = QLabel(
                "Left-click: add point \u00b7 Right-click: finish profile "
                "\u00b7 Double-click: done \u00b7 Z: undo \u00b7 Shift-click: delete"
            )
            hints.setStyleSheet("color: #555; padding: 4px;")

            # -- buttons --
            load_btn = QPushButton("Load")
            load_btn.setFixedWidth(80)
            load_btn.setToolTip("Load profiles from file (GeoPackage, Shapefile, ...)")
            load_btn.clicked.connect(self._on_load)

            save_btn = QPushButton("Save")
            save_btn.setFixedWidth(80)
            save_btn.setToolTip("Save finished profiles to file")
            save_btn.clicked.connect(self._on_save)

            done_btn = QPushButton("Done")
            done_btn.setFixedWidth(80)
            done_btn.clicked.connect(self._on_done)

            # -- layout --
            bottom = QHBoxLayout()
            bottom.addWidget(hints, stretch=1)
            bottom.addWidget(self._coord_label)
            bottom.addWidget(load_btn)
            bottom.addWidget(save_btn)
            bottom.addWidget(done_btn)

            # canvas + profile list side by side
            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.addWidget(canvas)
            splitter.addWidget(list_widget)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 0)

            layout = QVBoxLayout()
            layout.addWidget(toolbar)
            layout.addWidget(splitter, stretch=1)
            layout.addLayout(bottom)

            central = QWidget()
            central.setLayout(layout)
            self.setCentralWidget(central)

        def _update_coords(self, x: float, y: float) -> None:
            self._coord_label.setText(f"({x:.4f}, {y:.4f})")

        def _refresh_profile_list(self) -> None:
            """Rebuild the checkbox list to match current profiles."""
            self._updating_list = True
            self._profile_list.clear()
            for i, (prof, ps) in enumerate(
                zip(self._logic.profiles, self._logic._finished_states)
            ):
                item = QListWidgetItem(prof.name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if ps.visible else Qt.CheckState.Unchecked
                )
                item.setData(Qt.ItemDataRole.UserRole, i)
                self._profile_list.addItem(item)
            self._updating_list = False

        def _on_item_toggled(self, item: QListWidgetItem) -> None:
            if self._updating_list:
                return
            idx = item.data(Qt.ItemDataRole.UserRole)
            visible = item.checkState() == Qt.CheckState.Checked
            self._logic.set_visibility(idx, visible)

        def _set_all_visible(self, visible: bool) -> None:
            self._logic.set_all_visible(visible)
            self._refresh_profile_list()

        def _on_load(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Load profiles",
                "",
                "Vector files (*.gpkg *.shp *.geojson);;All files (*)",
            )
            if not path:
                return
            loaded = profiles_from_file(self._logic._data, path)
            if loaded:
                self._logic._load_profiles(loaded)
                self._logic._redraw()

        def _on_save(self) -> None:
            profs = self._logic.profiles
            if not profs:
                return
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save profiles",
                "profiles.gpkg",
                "GeoPackage (*.gpkg);;Shapefile (*.shp);;GeoJSON (*.geojson)",
            )
            if path:
                to_gdf(profs).to_file(path)

        def _confirm_close(self) -> bool:
            """Ask for confirmation if there are profiles that could be lost."""
            has_work = bool(self._logic.profiles) or any(
                len(ps.coords) >= 2
                for ps in self._logic._profile_states
                if not ps.finished
            )
            if not has_work:
                return True
            reply = QMessageBox.question(
                self,
                "Close window?",
                "Profiles have not been saved. Close anyway?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Save:
                self._on_save()
                return True
            return reply == QMessageBox.StandardButton.Discard

        def _do_finish_and_close(self) -> None:
            """Finish drawing and close without confirmation."""
            self._logic._do_finish()
            self.close()

        def _on_done(self) -> None:
            if self._confirm_close():
                self._do_finish_and_close()

        def closeEvent(self, event) -> None:  # noqa: N802
            if self._logic.done:
                super().closeEvent(event)
                return
            if self._confirm_close():
                self._logic._do_finish()
                super().closeEvent(event)
            else:
                event.ignore()

    app = QApplication.instance()
    created_app = app is None
    if created_app:
        app = QApplication(sys.argv)

    fig = Figure(figsize=(14, 5), constrained_layout=True)
    logic = _ProfileDrawingLogic(data, cmap, fig, profiles=profiles)

    window = _ProfileWindow(logic)

    # Double-click triggers _finish_drawing → route through confirmation
    logic.on_finish_requested = window._on_done

    window.show()
    window._canvas.setFocus()

    if created_app:
        app.exec()
    else:
        loop = QEventLoop()
        window.destroyed.connect(loop.quit)
        loop.exec()

    return logic.profiles
