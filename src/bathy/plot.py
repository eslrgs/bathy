"""Bathymetry visualisation functions."""

import cmocean.cm as cmo
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure

from bathy.analysis import (
    _GEOMORPHON_COLORS,
    _GEOMORPHON_LABELS,
    _clean_values,
    aspect,
    bpi,
    curvature,
    geomorphons,
    hypsometric_curve,
    rugosity,
    slope,
)


def get_extent(data: xr.DataArray) -> list[float]:
    """
    Get extent for matplotlib imshow.

    Parameters
    ----------
    data : xr.DataArray
        Data array with lon and lat coordinates

    Returns
    -------
    list[float]
        Extent as [lon_min, lon_max, lat_min, lat_max]
    """
    return [
        float(data.lon.min()),
        float(data.lon.max()),
        float(data.lat.min()),
        float(data.lat.max()),
    ]


def _hillshade(
    data: xr.DataArray, azimuth: float = 315, altitude: float = 45
) -> np.ndarray:
    """Compute hillshade using Horn's method (operates on negated elevations)."""
    gy, gx = np.gradient(-data.values.astype(float))
    az_rad = np.radians(360 - azimuth + 90)
    alt_rad = np.radians(altitude)
    shaded = (
        np.sin(alt_rad) + np.cos(alt_rad) * (np.cos(az_rad) * gx + np.sin(az_rad) * gy)
    ) / np.sqrt(1 + gx**2 + gy**2)
    return np.clip(shaded, 0, 1)


def _add_contours(
    data: xr.DataArray, ax: "Axes", contours: int | list[float], **kwargs
) -> None:
    """Add contour lines to an existing axes."""
    cs = data.plot.contour(
        ax=ax,
        levels=contours,
        colors="black",
        alpha=0.8,
        linewidths=1,
        linestyles="-",
        **kwargs,
    )
    ax.clabel(cs, inline=True, fontsize=8)


def _plot_grid(
    values: np.ndarray,
    data: xr.DataArray,
    cmap,
    label: str,
    contours: int | list[float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plot a 2-D grid with colorbar, optional contours, and axis labels."""
    extent = get_extent(data)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        values,
        cmap=cmap,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )
    plt.colorbar(im, ax=ax, label=label)
    if contours is not None:
        _add_contours(data, ax, contours)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    return fig, ax


def plot_bathy(
    data: xr.DataArray,
    contours: int | list[float] | None = None,
    cmap=None,
    mask_land: bool = True,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    Plot bathymetry elevation.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    contours : int or list[float], optional
        Number of contour levels or specific levels (in metres)
    cmap : str or Colormap, optional
        Colormap. Defaults to cmocean 'deep_r'.
    mask_land : bool, default True
        If True, mask positive elevations (land).
    **kwargs
        Additional arguments passed to xarray plot

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes for further customisation.
    """
    if cmap is None:
        cmap = cmo.deep_r

    fig, ax = plt.subplots(figsize=(10, 8))

    data_masked = data.where(data < 0) if mask_land else data

    if "cbar_kwargs" not in kwargs:
        kwargs["cbar_kwargs"] = {"label": "Elevation (m)"}

    data_masked.plot(ax=ax, cmap=cmap, **kwargs)

    if contours is not None:
        _add_contours(data, ax, contours)

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    return fig, ax


def plot_hillshade(
    data: xr.DataArray,
    azimuth: float = 315,
    altitude: float = 45,
    contours: int | list[float] | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    Create hillshade visualisation.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    azimuth : float, default 315
        Light source azimuth in degrees
    altitude : float, default 45
        Light source altitude in degrees
    contours : int or list[float], optional
        Contour levels
    **kwargs
        Additional arguments passed to imshow

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes for further customisation.
    """
    shaded = _hillshade(data, azimuth=azimuth, altitude=altitude)
    extent = get_extent(data)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(
        shaded, cmap="gray", origin="lower", extent=extent, aspect="auto", **kwargs
    )

    if contours is not None:
        _add_contours(data, ax, contours)

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    return fig, ax


def plot_slope(
    data: xr.DataArray,
    contours: int | list[float] | None = None,
    vmax: float | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    Plot seafloor slope.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    contours : int or list[float], optional
        Contour levels
    vmax : float, optional
        Maximum slope value for colour scale.
    **kwargs
        Additional arguments passed to imshow

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes for further customisation.
    """
    slope_data = slope(data)
    if vmax is None:
        vmax = float(np.nanpercentile(slope_data.values, 99))
    return _plot_grid(
        slope_data.values,
        data,
        "Greys",
        "Slope (°)",
        contours=contours,
        vmin=0,
        vmax=vmax,
        **kwargs,
    )


def plot_curvature(
    data: xr.DataArray,
    contours: int | list[float] | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    Plot seafloor curvature.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    contours : int or list[float], optional
        Contour levels
    **kwargs
        Additional arguments passed to imshow

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes for further customisation.
    """
    curvature_data = curvature(data)
    vmax = np.nanmax(np.abs(curvature_data.values))
    return _plot_grid(
        curvature_data.values,
        data,
        cmo.balance,
        "Curvature",
        contours=contours,
        vmin=-vmax,
        vmax=vmax,
        **kwargs,
    )


def plot_bpi(
    data: xr.DataArray,
    radius_km: float = 1.0,
    contours: int | list[float] | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    Plot Bathymetric Position Index (BPI).

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    radius_km : float, default 1.0
        Neighbourhood radius in kilometres
    contours : int or list[float], optional
        Contour levels
    **kwargs
        Additional arguments passed to imshow

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes for further customisation.
    """
    bpi_data = bpi(data, radius_km)
    vmax = np.nanmax(np.abs(bpi_data.values))
    return _plot_grid(
        bpi_data.values,
        data,
        cmo.balance,
        f"BPI (r={radius_km} km)",
        contours=contours,
        vmin=-vmax,
        vmax=vmax,
        **kwargs,
    )


def plot_rugosity(
    data: xr.DataArray,
    radius_km: float = 1.0,
    contours: int | list[float] | None = None,
    vmax: float | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    Plot Vector Ruggedness Measure (VRM).

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    radius_km : float, default 1.0
        Neighbourhood radius in kilometres
    contours : int or list[float], optional
        Contour levels
    vmax : float, optional
        Maximum VRM value for colour scale.
    **kwargs
        Additional arguments passed to imshow

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes for further customisation.
    """
    rug_data = rugosity(data, radius_km)
    if vmax is None:
        vmax = float(np.nanpercentile(rug_data.values, 99))
    return _plot_grid(
        rug_data.values,
        data,
        cmo.amp,
        f"Rugosity VRM (r={radius_km} km)",
        contours=contours,
        vmin=0,
        vmax=vmax,
        **kwargs,
    )


def plot_aspect(
    data: xr.DataArray,
    contours: int | list[float] | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    Plot seafloor aspect.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    contours : int or list[float], optional
        Contour levels
    **kwargs
        Additional arguments passed to imshow

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes for further customisation.
    """
    asp_data = aspect(data)
    extent = get_extent(data)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        asp_data.values,
        cmap=cmo.phase,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=0,
        vmax=360,
        **kwargs,
    )
    cbar = plt.colorbar(im, ax=ax, label="Aspect")
    cbar.set_ticks([0, 90, 180, 270, 360])
    cbar.set_ticklabels(["0° N", "90° E", "180° S", "270° W", "360° N"])

    if contours is not None:
        _add_contours(data, ax, contours)

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    return fig, ax


def plot_geomorphons(
    data: xr.DataArray,
    lookup_km: float = 2.0,
    flatness_threshold: float = 1.0,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    Plot seafloor morphology using geomorphons.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    lookup_km : float, default 2.0
        Lookup distance in kilometres.
    flatness_threshold : float, default 1.0
        Flatness angle threshold in degrees.
    **kwargs
        Additional arguments passed to imshow.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes for further customisation.

    Examples
    --------
    >>> plot_geomorphons(data, lookup_km=2.0)
    """
    geom_data = geomorphons(data, lookup_km, flatness_threshold)
    extent = get_extent(data)

    cmap = ListedColormap(_GEOMORPHON_COLORS)
    norm = BoundaryNorm(np.arange(0.5, 11.5), len(_GEOMORPHON_COLORS))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        geom_data.values,
        cmap=cmap,
        norm=norm,
        origin="lower",
        extent=extent,
        aspect="auto",
        **kwargs,
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks(range(1, 11))
    cbar.set_ticklabels(_GEOMORPHON_LABELS)

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    return fig, ax


def plot_overview(
    data: xr.DataArray,
    bpi_radius_km: float = 1.0,
    rugosity_radius_km: float = 1.0,
    geomorphons_lookup_km: float = 2.0,
    label_prefix: list[str] | None = None,
) -> tuple[Figure, np.ndarray]:
    """
    Plot key bathymetric metrics as subplot overview.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    bpi_radius_km : float, default 1.0
        Neighbourhood radius for BPI in kilometres.
    rugosity_radius_km : float, default 1.0
        Neighbourhood radius for rugosity in kilometres.
    geomorphons_lookup_km : float, default 2.0
        Lookup distance for geomorphons in kilometres.
    label_prefix : list[str] | None, default None
        Optional list of 8 prefix strings for panel titles.

    Examples
    --------
    Returns
    -------
    tuple[Figure, np.ndarray]
        Matplotlib figure and array of axes for further customisation.

    Examples
    --------
    >>> plot_overview(data)
    """
    n_panels = 8
    prefixes = label_prefix or [""] * n_panels
    if len(prefixes) != n_panels:
        raise ValueError(
            f"label_prefix must have {n_panels} entries, got {len(prefixes)}"
        )

    def title(i: int, name: str) -> str:
        p = prefixes[i]
        return f"{p} {name}" if p else name

    extent = get_extent(data)
    imkw = dict(origin="lower", extent=extent, aspect="auto")

    hs = _hillshade(data)
    sl = slope(data)
    asp = aspect(data)
    cu = curvature(data)
    bp = bpi(data, bpi_radius_km)
    vr = rugosity(data, rugosity_radius_km)
    gm = geomorphons(data, geomorphons_lookup_km)

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(12, 20),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    im = axes[0, 0].imshow(data.where(data < 0).values, cmap=cmo.deep_r, **imkw)
    plt.colorbar(im, ax=axes[0, 0], label="m")
    axes[0, 0].set_title(title(0, "Bathymetry"))

    axes[0, 1].imshow(hs, cmap="gray", **imkw)
    axes[0, 1].set_title(title(1, "Hillshade"))

    vmax = float(np.nanpercentile(sl.values, 99))
    im = axes[1, 0].imshow(sl.values, cmap="Greys", vmin=0, vmax=vmax, **imkw)
    plt.colorbar(im, ax=axes[1, 0], label="°")
    axes[1, 0].set_title(title(2, "Slope"))

    im = axes[1, 1].imshow(asp.values, cmap=cmo.phase, vmin=0, vmax=360, **imkw)
    cbar = plt.colorbar(im, ax=axes[1, 1])
    cbar.set_ticks([0, 90, 180, 270, 360])
    cbar.set_ticklabels(["N", "E", "S", "W", "N"])
    axes[1, 1].set_title(title(3, "Aspect"))

    vmax = float(np.nanpercentile(np.abs(cu.values), 99))
    im = axes[2, 0].imshow(cu.values, cmap=cmo.balance, vmin=-vmax, vmax=vmax, **imkw)
    plt.colorbar(im, ax=axes[2, 0], label="m⁻¹")
    axes[2, 0].set_title(title(4, "Curvature"))

    vmax = float(np.nanpercentile(np.abs(bp.values), 99))
    im = axes[2, 1].imshow(bp.values, cmap=cmo.balance, vmin=-vmax, vmax=vmax, **imkw)
    plt.colorbar(im, ax=axes[2, 1], label="m")
    axes[2, 1].set_title(title(5, f"BPI (r={bpi_radius_km} km)"))

    vmax = float(np.nanpercentile(vr.values, 99))
    im = axes[3, 0].imshow(vr.values, cmap=cmo.amp, vmin=0, vmax=vmax, **imkw)
    plt.colorbar(im, ax=axes[3, 0], label="VRM")
    axes[3, 0].set_title(title(6, f"Rugosity (r={rugosity_radius_km} km)"))

    im = axes[3, 1].imshow(
        gm.values,
        cmap=ListedColormap(_GEOMORPHON_COLORS),
        norm=BoundaryNorm(np.arange(0.5, 11.5), 10),
        **imkw,
    )
    cbar = plt.colorbar(im, ax=axes[3, 1])
    cbar.set_ticks(range(1, 11))
    cbar.set_ticklabels(_GEOMORPHON_LABELS, fontsize=7)
    axes[3, 1].set_title(title(7, f"Geomorphons ({geomorphons_lookup_km} km)"))

    for ax in axes[3, :]:
        ax.set_xlabel("Longitude (°)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Latitude (°)")
    for ax in axes.ravel():
        ax.label_outer()

    return fig, axes


def plot_histogram(data: xr.DataArray, bins: int = 50, **kwargs) -> tuple[Figure, Axes]:
    """
    Plot histogram of elevation values.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes for further customisation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    values = _clean_values(data)

    ax.hist(values, bins=bins, edgecolor="black", **kwargs)
    ax.axvline(0, color="blue", linestyle="--", linewidth=2, label="Sea level")
    ax.set_xlabel("Elevation (m)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_depth_zones(
    data: xr.DataArray,
    zones: list[float] | None = None,
    labels: list[str] | None = None,
    contours: int | list[float] | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    Plot bathymetry color-coded by depth zones.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    zones : list[float], optional
        Depth boundaries (default: [0, -200, -1000, -4000])
    labels : list[str], optional
        Zone labels (default: ['Shelf', 'Slope', 'Abyss', 'Deep'])
    contours : int or list[float], optional
        Contour levels
    **kwargs
        Additional arguments passed to imshow

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes for further customisation.
    """
    if zones is None:
        zones = [0, -200, -1000, -4000]
    if labels is None:
        labels = ["Shelf", "Slope", "Abyss", "Deep"]

    sorted_zones = sorted(zones)
    n_zones = len(sorted_zones)

    boundaries = [data.min().values] + sorted_zones
    reversed_labels = labels[::-1]

    deep_colors = cmo.deep(np.linspace(1, 0, n_zones))
    colors = ListedColormap(deep_colors)
    norm = BoundaryNorm(boundaries, n_zones)

    extent = get_extent(data)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        data.values,
        cmap=colors,
        norm=norm,
        origin="lower",
        extent=extent,
        aspect="auto",
        **kwargs,
    )

    if contours is not None:
        _add_contours(data, ax, contours)

    cbar = plt.colorbar(im, ax=ax, label="Depth zone")

    tick_positions = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(n_zones)]
    tick_labels = [
        f"{reversed_labels[i]}\n({int(boundaries[i + 1])} to {int(boundaries[i])} m)"
        for i in range(n_zones)
    ]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    return fig, ax


def plot_surface3d(
    data: xr.DataArray,
    stride: int = 10,
    vertical_exaggeration: float = 50.0,
    smooth: int | None = None,
    elev: float = 30,
    azim: float = -60,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    Create static 3D surface plot.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    stride : int, default 10
        Stride for downsampling (every Nth point)
    vertical_exaggeration : float, default 50.0
        Factor to exaggerate the vertical scale.
    smooth : int, optional
        Uniform filter kernel size for smoothing.
    elev : float, default 30
        Elevation viewing angle in degrees.
    azim : float, default -60
        Azimuth viewing angle in degrees.
    **kwargs
        Additional arguments passed to plot_surface

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes for further customisation.
    """
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection="3d")

    lon = data.lon.values[::stride]
    lat = data.lat.values[::stride]
    z = data.values[::stride, ::stride]

    if smooth is not None:
        from scipy.ndimage import uniform_filter  # noqa: PLC0415

        z = uniform_filter(z, size=smooth, mode="nearest")

    lon_grid, lat_grid = np.meshgrid(lon, lat)

    surf = ax.plot_surface(
        lon_grid,
        lat_grid,
        z,
        cmap=cmo.deep_r,
        linewidth=0,
        antialiased=True,
        **kwargs,
    )
    fig.colorbar(surf, ax=ax, label="Elevation (m)", shrink=0.5, pad=0.1)

    lat_centre = float(data.lat.mean())
    lon_scale = np.cos(np.radians(lat_centre))
    ax.set_box_aspect(
        [
            (lon.max() - lon.min()) * lon_scale,
            lat.max() - lat.min(),
            (z.max() - z.min()) * vertical_exaggeration / 1000,
        ]
    )

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_zlabel("Elevation (m)")
    plt.tight_layout()
    return fig, ax


def plot_hypsometric_curve(
    data: xr.DataArray, bins: int = 100, **kwargs
) -> tuple[Figure, Axes]:
    """
    Plot the hypsometric curve.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    bins : int, default 100
        Number of elevation bins
    **kwargs
        Additional arguments passed to plt.plot

    Examples
    --------
    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes for further customisation.

    Examples
    --------
    >>> plot_hypsometric_curve(data)
    """
    rel_area, rel_elev = hypsometric_curve(data, bins)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(rel_area, rel_elev, linewidth=2, **kwargs)
    ax.plot([0, 1], [1, 0], "k--", alpha=0.3, label="Equidimensional")
    ax.legend()
    ax.set_xlabel("Relative Area (a/A)")
    ax.set_ylabel("Relative Elevation (h/H)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    return fig, ax
