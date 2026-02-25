"""Interactive profile drawing on bathymetry maps."""

import cmocean.cm as cmo
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.backend_bases import MouseButton

from bathy.plot import get_extent
from bathy.profile import profile_from_coordinates

_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def _style_profile_axis(ax, n_profiles):
    """Apply consistent labels and styling to the profile axis."""
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"{n_profiles} profile{'s' if n_profiles != 1 else ''}")
    ax.grid(True, alpha=0.3)
    if n_profiles:
        ax.legend()


def draw_profile(
    data: xr.DataArray,
    cmap=cmo.deep_r,
) -> dict:
    """
    Interactively draw profiles on a bathymetry map.

    Left-click to add waypoints for the current profile.
    Right-click to finish the current profile and start a new one.
    Double-click to finish drawing entirely.
    Requires the ipympl backend (``%matplotlib widget``) in Jupyter.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data with lon and lat coordinates.
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
        # Double-click to stop drawing
        # Then access profiles:
        result["profiles"]       # list of all profiles
        result["profiles"][0]    # first profile
    """
    extent = get_extent(data)

    fig, (ax_map, ax_profile) = plt.subplots(1, 2, figsize=(14, 5))
    ax_map.imshow(
        data.values,
        cmap=cmap,
        origin="lower",
        extent=extent,
        aspect="auto",
    )
    ax_map.set_xlabel("Longitude (°)")
    ax_map.set_ylabel("Latitude (°)")
    ax_map.set_title("Draw profile (right-click: finish, double-click: done)")

    ax_profile.set_xlabel("Distance (km)")
    ax_profile.set_ylabel("Elevation (m)")
    ax_profile.set_title("Profile")
    ax_profile.grid(True, alpha=0.3)

    state = {"profiles": [], "done": False}
    coords = []
    color = _COLORS[0]
    (line,) = ax_map.plot([], [], ".-", color=color, linewidth=2, markersize=8)

    def _finish_current():
        if len(coords) < 2:
            return
        idx = len(state["profiles"])
        name = f"Profile {idx + 1}"
        prof = profile_from_coordinates(data, coords, name=name)
        state["profiles"].append(prof)

        plot_color = _COLORS[idx % len(_COLORS)]
        lons = [pt[0] for pt in coords]
        lats = [pt[1] for pt in coords]
        ax_map.plot(lons, lats, "-", color=plot_color, linewidth=2, label=name)

        ax_profile.clear()
        for i, p in enumerate(state["profiles"]):
            ax_profile.plot(
                p.distances,
                p.elevations,
                color=_COLORS[i % len(_COLORS)],
                label=p.name,
            )
        _style_profile_axis(ax_profile, len(state["profiles"]))

        coords.clear()

    def _on_click(event):
        if event.inaxes != ax_map or state["done"]:
            return

        if event.dblclick:
            _finish_current()
            state["done"] = True
            n = len(state["profiles"])
            ax_map.set_title(f"{n} profile{'s' if n != 1 else ''} extracted")
            ax_map.legend()
            line.set_data([], [])
            fig.canvas.draw_idle()
            return

        if event.button == MouseButton.RIGHT:
            _finish_current()
            next_color = _COLORS[len(state["profiles"]) % len(_COLORS)]
            line.set_color(next_color)
            line.set_data([], [])
            fig.canvas.draw_idle()
            return

        if event.button == MouseButton.LEFT:
            coords.append((event.xdata, event.ydata))
            lons = [pt[0] for pt in coords]
            lats = [pt[1] for pt in coords]
            line.set_data(lons, lats)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", _on_click)
    return state
