"""Profile plotting functions."""

import logging

import cmocean.cm as cmo
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter1d

from bathy.plot import get_extent
from bathy.profile import (
    Profile,
    _ensure_descending,
    _normalise_profile,
    gradient,
    knickpoints,
)
from bathy.utils import axis_labels, crs_axis_labels

logger = logging.getLogger(__name__)


def plot_profile(
    profile: Profile,
    show_map: bool = False,
    smooth: float | None = None,
    normalize: bool = False,
    ensure_descending: bool = False,
    cmap=cmo.deep_r,
    bathymetry_data: xr.DataArray | None = None,
    **kwargs,
) -> tuple[Figure, list[Axes]]:
    """
    Plot the bathymetric profile.

    Parameters
    ----------
    profile : Profile
    show_map : bool
        If True, show map with profile line. Requires bathymetry_data.
    smooth : float, optional
        Gaussian smoothing sigma.
    normalize : bool
        If True, normalize elevation and distance to 0-1.
    ensure_descending : bool
        If True, orient profile to descend from higher to lower elevation.
    bathymetry_data : xr.DataArray, optional
        Background data for map view (required when show_map=True).
    **kwargs
        Additional arguments passed to matplotlib plot()

    Returns
    -------
    tuple[Figure, list[Axes]]
        Figure and list of axes (one element without map, two with map).
    """
    elevations = (
        gaussian_filter1d(profile.elevations, sigma=smooth)
        if smooth
        else profile.elevations
    )
    distances = profile.distances / 1000

    if ensure_descending:
        distances, elevations = _ensure_descending(distances, elevations)

    if normalize:
        distances, elevations = _normalise_profile(distances, elevations)

    ylim = (float(elevations.min()), float(elevations.max()))
    xlim = (float(distances.min()), float(distances.max()))

    if show_map:
        fig, (ax_map, ax_profile) = plt.subplots(1, 2, figsize=(16, 6))

        if bathymetry_data is not None:
            extent = get_extent(bathymetry_data)
            ax_map.imshow(
                bathymetry_data.values,
                cmap=cmap,
                origin="lower",
                extent=extent,
                aspect="auto",
            )
        path_xs = profile.metadata.get("path_xs", [profile.start_x, profile.end_x])
        path_ys = profile.metadata.get("path_ys", [profile.start_y, profile.end_y])
        ax_map.plot(path_xs, path_ys, "r-", linewidth=2, label="Profile line")
        ax_map.plot(path_xs[0], path_ys[0], "go", markersize=10, label="Start")
        ax_map.plot(path_xs[-1], path_ys[-1], "ro", markersize=10, label="End")
        if bathymetry_data is not None:
            x_label, y_label = axis_labels(bathymetry_data)
        else:
            x_label, y_label = crs_axis_labels(profile.crs)
        ax_map.set_xlabel(x_label)
        ax_map.set_ylabel(y_label)
        ax_map.legend()
    else:
        fig, ax_profile = plt.subplots(figsize=(12, 5))

    ax_profile.plot(distances, elevations, **kwargs)
    ax_profile.fill_between(distances, elevations, elevations.min(), alpha=0.3)

    ax_profile.set_xlabel("Normalized distance" if normalize else "Distance (km)")
    ax_profile.set_ylabel("Normalized elevation" if normalize else "Elevation (m)")
    ax_profile.set_xlim(xlim)
    ax_profile.set_ylim(ylim)
    ax_profile.grid(True, alpha=0.3)

    if show_map:
        return fig, [ax_map, ax_profile]
    return fig, [ax_profile]


def plot_knickpoints(
    profile: Profile,
    knickpoints_df: pl.DataFrame | None = None,
    threshold: float | None = None,
    smooth: float | None = None,
    **kwargs,
) -> tuple[Figure, list[Axes]]:
    """
    Plot profile with knickpoints marked.

    Parameters
    ----------
    profile : Profile
    knickpoints_df : pl.DataFrame, optional
        Knickpoint data from knickpoints(). Detected if None.
    threshold : float, optional
        Minimum slope break for detection (ignored if knickpoints_df provided).
    smooth : float, optional
        Smoothing sigma (ignored if knickpoints_df provided).
    **kwargs
        Additional arguments passed to plot_profile

    Returns
    -------
    tuple[Figure, list[Axes]]
        Figure and list of axes.
    """
    if knickpoints_df is None:
        knickpoints_df = knickpoints(profile, threshold=threshold, smooth=smooth)

    fig, axes = plot_profile(profile, **kwargs)

    if len(knickpoints_df) == 0:
        logger.info("No knickpoints detected. Try adjusting threshold or smoothing.")
        return fig, axes

    axes[-1].scatter(
        knickpoints_df["distance_m"] / 1000,
        knickpoints_df["depth_m"],
        c="red",
        s=50,
        zorder=5,
        label="Knickpoints",
    )
    axes[-1].legend()
    return fig, axes


def plot_gradient(profile: Profile, **kwargs) -> tuple[Figure, list[Axes]]:
    """
    Plot the gradient (derivative) along the profile.

    Parameters
    ----------
    profile : Profile
    **kwargs
        Additional arguments passed to plot

    Returns
    -------
    tuple[Figure, list[Axes]]
        Figure and list containing the single gradient axes.
    """
    grad = gradient(profile)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(profile.distances / 1000, grad, **kwargs)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Slope (°)")
    ax.grid(True, alpha=0.3)

    return fig, [ax]


def plot_canyons(
    profile: Profile,
    canyons: pl.DataFrame,
    **kwargs,
) -> tuple[Figure, list[Axes]]:
    """
    Plot profile with canyons marked.

    Parameters
    ----------
    profile : Profile
    canyons : pl.DataFrame
        Canyon data from ``get_canyons()``.
    **kwargs
        Additional arguments passed to plot_profile()

    Returns
    -------
    tuple[Figure, list[Axes]]
        Figure and list of axes.
    """

    if len(canyons) == 0:
        logger.info("No canyons detected. Try adjusting prominence or smoothing.")
        return plot_profile(profile, **kwargs)

    fig, axes = plot_profile(profile, **kwargs)
    ax = axes[-1]

    for row in canyons.iter_rows(named=True):
        floor_km = row["floor_distance"] / 1000
        floor_elev = row["floor_elevation"]
        shoulder_elev = row["shoulder_elevation"]
        ws_km, we_km = row["width_start"] / 1000, row["width_end"] / 1000

        ax.plot(floor_km, floor_elev, "ro", markersize=8, zorder=10)
        ax.plot(
            [ws_km, we_km],
            [shoulder_elev] * 2,
            "k--",
            linewidth=1.5,
            alpha=0.7,
            zorder=5,
        )
        ax.plot(
            [floor_km] * 2,
            [floor_elev, shoulder_elev],
            "k--",
            linewidth=1.5,
            alpha=0.7,
            zorder=5,
        )

    return fig, axes


def plot_profiles(
    profiles: Profile | list[Profile],
    show_map: bool = False,
    normalize: bool = False,
    ensure_descending: bool = False,
    bathymetry_data: xr.DataArray | None = None,
    cmap=cmo.deep_r,
    **kwargs,
) -> tuple[Figure, list[Axes]]:
    """
    Plot multiple profiles on the same axes.

    Parameters
    ----------
    profiles : Profile or list[Profile]
    show_map : bool
        If True, show map with profile lines alongside the profile plot.
        Requires bathymetry_data.
    normalize : bool
        If True, normalize each profile's elevation and distance to 0-1.
    ensure_descending : bool
        If True, orient profiles to descend from higher to lower elevation.
    bathymetry_data : xr.DataArray, optional
        Background data for map view.
    **kwargs
        Additional arguments passed to matplotlib plot()

    Returns
    -------
    tuple[Figure, list[Axes]]
        Figure and list of axes (one element without map, two with map).

    Examples
    --------
    >>> from bathy.profile_plot import plot_profiles
    >>> prof1 = extract_profile(data, start=(-8, 52), end=(-2, 58), name="Profile 1")
    >>> prof2 = extract_profile(data, start=(-8, 53), end=(-2, 59), name="Profile 2")
    >>> plot_profiles([prof1, prof2])
    """
    if isinstance(profiles, Profile):
        profiles = [profiles]

    if not profiles:
        raise ValueError("Need at least one profile to plot")

    if show_map:
        fig, (ax_map, ax_profile) = plt.subplots(1, 2, figsize=(16, 6))

        if bathymetry_data is not None:
            extent = get_extent(bathymetry_data)
            ax_map.imshow(
                bathymetry_data.values,
                cmap=cmap,
                origin="lower",
                extent=extent,
                aspect="auto",
                alpha=0.6,
            )

        for i, prof in enumerate(profiles, start=1):
            label = prof.name if prof.name else f"Profile {i}"
            path_xs = prof.metadata.get("path_xs", [prof.start_x, prof.end_x])
            path_ys = prof.metadata.get("path_ys", [prof.start_y, prof.end_y])
            ax_map.plot(path_xs, path_ys, "-", linewidth=2, label=label)
            ax_map.plot(path_xs[0], path_ys[0], "o", markersize=6)
            ax_map.plot(path_xs[-1], path_ys[-1], "s", markersize=6)

        if bathymetry_data is not None:
            x_label, y_label = axis_labels(bathymetry_data)
        else:
            x_label, y_label = crs_axis_labels(profiles[0].crs)
        ax_map.set_xlabel(x_label)
        ax_map.set_ylabel(y_label)
        ax_map.legend()
    else:
        fig, ax_profile = plt.subplots(figsize=(12, 6))

    for i, prof in enumerate(profiles, start=1):
        distances = prof.distances / 1000
        elevations = prof.elevations.copy()

        if ensure_descending:
            distances, elevations = _ensure_descending(distances, elevations)

        if normalize:
            distances, elevations = _normalise_profile(distances, elevations)

        label = prof.name if prof.name else f"Profile {i}"
        ax_profile.plot(distances, elevations, label=label, **kwargs)

    ax_profile.set_xlabel("Normalized distance" if normalize else "Distance (km)")
    ax_profile.set_ylabel("Normalized elevation" if normalize else "Elevation (m)")
    ax_profile.grid(True, alpha=0.3)
    ax_profile.legend()

    if show_map:
        return fig, [ax_map, ax_profile]
    return fig, [ax_profile]


def plot_profiles_grid(
    profiles: Profile | list[Profile],
    cols: int = 2,
    figsize: tuple[float, float] | None = None,
    main_profile: Profile | None = None,
    smooth: float | None = None,
    normalize: bool = False,
    ensure_descending: bool = False,
    **kwargs,
) -> tuple[Figure, np.ndarray]:
    """
    Plot multiple profiles in a grid of subplots.

    Parameters
    ----------
    profiles : Profile or list[Profile]
    cols : int
        Number of columns in the grid (default: 2)
    figsize : tuple[float, float], optional
        Figure size. Calculated if None.
    main_profile : Profile, optional
        Main profile; marks intersection with each cross-section.
    smooth : float, optional
        Gaussian smoothing sigma.
    normalize : bool
        If True, normalize each profile's elevation and distance to 0-1.
    ensure_descending : bool
        If True, orient profiles to descend from higher to lower elevation.
    **kwargs
        Additional arguments passed to matplotlib plot()

    Returns
    -------
    Figure, np.ndarray

    Examples
    --------
    >>> from bathy.profile_plot import plot_profiles_grid
    >>> profiles = profiles_from_file(data, "canyons.shp")
    >>> plot_profiles_grid(profiles[:10])
    """
    if isinstance(profiles, Profile):
        profiles = [profiles]

    if not profiles:
        raise ValueError("Need at least one profile to plot")

    n_profiles = len(profiles)
    rows = (n_profiles + cols - 1) // cols

    if figsize is None:
        figsize = (7 * cols, 3.5 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for i, prof in enumerate(profiles):
        ax = axes[i]

        elevations = (
            gaussian_filter1d(prof.elevations, sigma=smooth)
            if smooth
            else prof.elevations.copy()
        )
        distances = prof.distances / 1000

        if ensure_descending:
            distances, elevations = _ensure_descending(distances, elevations)

        if normalize:
            distances, elevations = _normalise_profile(distances, elevations)

        ylim = (float(elevations.min()), float(elevations.max()))
        xlim = (float(distances.min()), float(distances.max()))

        ax.plot(distances, elevations, **kwargs)
        ax.fill_between(distances, elevations, elevations.min(), alpha=0.3)

        if main_profile is not None:
            mid_distance = distances[len(distances) // 2]
            ax.axvline(
                mid_distance,
                color="black",
                linestyle="-",
                linewidth=1.5,
                alpha=0.7,
                zorder=10,
            )

        ax.set_xlabel("Normalized distance" if normalize else "Distance (km)")
        ax.set_ylabel("Normalized elevation" if normalize else "Elevation (m)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        title = prof.name if prof.name else f"Profile {i + 1}"
        ax.set_title(f"{title} ({prof.distances[-1] / 1000:.1f} km)")
        ax.grid(True, alpha=0.3)

    for i in range(n_profiles, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig, axes


def plot_profiles_map(
    profiles: Profile | list[Profile],
    bathymetry_data: xr.DataArray | None = None,
    main_profile: Profile | None = None,
    cmap=cmo.deep_r,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    Plot profile locations on a map.

    Parameters
    ----------
    profiles : Profile or list[Profile]
    bathymetry_data : xr.DataArray, optional
        Background bathymetry data.
    main_profile : Profile, optional
        Main profile to highlight.
    **kwargs
        Additional arguments passed to matplotlib plot()

    Returns
    -------
    Figure, Axes

    Examples
    --------
    >>> from bathy.profile_plot import plot_profiles_map
    >>> plot_profiles_map([prof1, prof2], bathymetry_data=data)
    """
    if isinstance(profiles, Profile):
        profiles = [profiles]

    if not profiles:
        raise ValueError("Need at least one profile to plot")

    fig, ax = plt.subplots(figsize=(10, 8))

    if bathymetry_data is not None:
        extent = get_extent(bathymetry_data)
        ax.imshow(
            bathymetry_data.values,
            cmap=cmap,
            origin="lower",
            extent=extent,
            aspect="auto",
            alpha=0.6,
        )

    for i, prof in enumerate(profiles, start=1):
        label = prof.name if prof.name else f"Profile {i}"

        if "path_xs" in prof.metadata and "path_ys" in prof.metadata:
            xs = prof.metadata["path_xs"]
            ys = prof.metadata["path_ys"]
            ax.plot(xs, ys, "-", linewidth=2, label=label, **kwargs)
            ax.plot(xs[0], ys[0], "o", markersize=8)
            ax.plot(xs[-1], ys[-1], "s", markersize=8)
        else:
            ax.plot(
                [prof.start_x, prof.end_x],
                [prof.start_y, prof.end_y],
                "-",
                linewidth=2,
                label=label,
                **kwargs,
            )
            ax.plot(prof.start_x, prof.start_y, "o", markersize=8)
            ax.plot(prof.end_x, prof.end_y, "s", markersize=8)

    if main_profile is not None:
        main_label = main_profile.name if main_profile.name else "Main Profile"
        ax.plot(
            [main_profile.start_x, main_profile.end_x],
            [main_profile.start_y, main_profile.end_y],
            "r-",
            linewidth=3,
            label=main_label,
            zorder=10,
        )
        ax.plot(
            main_profile.start_x,
            main_profile.start_y,
            "go",
            markersize=10,
            zorder=11,
            label="Start",
        )
        ax.plot(
            main_profile.end_x,
            main_profile.end_y,
            "rs",
            markersize=10,
            zorder=11,
            label="End",
        )

    if bathymetry_data is not None:
        x_label, y_label = axis_labels(bathymetry_data)
    else:
        x_label, y_label = crs_axis_labels(profiles[0].crs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if any(p.name for p in profiles) or main_profile is not None:
        ax.legend()

    return fig, ax
