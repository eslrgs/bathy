"""Bathymetry analysis functions."""

from __future__ import annotations

from collections.abc import Sequence

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xarray as xr
from geographiclib.geodesic import Geodesic
from shapely.geometry import LineString

from bathy.utils import get_crs, get_dim_names, is_projected

_GEOMORPHON_LABELS = [
    "Flat",
    "Peak",
    "Ridge",
    "Shoulder",
    "Spur",
    "Slope",
    "Hollow",
    "Footslope",
    "Valley",
    "Pit",
]
_GEOMORPHON_COLORS = [
    "#d9d9d9",
    "#7a0000",
    "#c0392b",
    "#e07b6b",
    "#e8a45a",
    "#909090",
    "#5dade2",
    "#2980b9",
    "#1a5276",
    "#0b2545",
]


# ============================================================================
# Internal helpers
# ============================================================================


def _cell_size_metres(data: xr.DataArray) -> tuple[float, float]:
    """Return (dy, dx) cell size in metres.

    For projected CRS the coordinate spacing is already in metres.
    For geographic CRS a geodesic measurement on WGS84 is used.
    """

    x_dim, y_dim = get_dim_names(data)

    if is_projected(data):
        dy = float(np.abs(np.diff(data[y_dim].values).mean()))
        dx = float(np.abs(np.diff(data[x_dim].values).mean()))
        return dy, dx

    lat_spacing = np.abs(np.diff(data[y_dim].values).mean())
    lon_spacing = np.abs(np.diff(data[x_dim].values).mean())
    lat_centre = float(data[y_dim].mean())
    lon_centre = float(data[x_dim].mean())

    geod = Geodesic.WGS84
    dy = geod.Inverse(lat_centre, lon_centre, lat_centre + lat_spacing, lon_centre)[
        "s12"
    ]
    dx = geod.Inverse(lat_centre, lon_centre, lat_centre, lon_centre + lon_spacing)[
        "s12"
    ]
    return dy, dx


def _clean_values(data: xr.DataArray) -> np.ndarray:
    """Get flattened array with NaN values removed.

    Returns an empty float64 array if all values are NaN.
    """
    values = data.values.ravel().astype(float)
    return values[~np.isnan(values)]


def _gradients(
    data: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return (gy, gx, dy, dx_rows) for gradient computation.

    For geographic CRS, east-west cell size is computed per latitude row
    to account for convergence of meridians.  For projected CRS, cell
    sizes are uniform.
    """

    x_dim, y_dim = get_dim_names(data)
    z = data.values.astype(float)

    if is_projected(data):
        dy = float(np.abs(np.diff(data[y_dim].values).mean()))
        dx = float(np.abs(np.diff(data[x_dim].values).mean()))
        dx_rows = np.full(z.shape[0], dx)
        gy = np.gradient(z, dy, axis=0)
        gx = np.gradient(z, dx, axis=1)
        return gy, gx, dy, dx_rows

    geod = Geodesic.WGS84
    lat_spacing = np.abs(np.diff(data[y_dim].values).mean())
    lon_spacing = np.abs(np.diff(data[x_dim].values).mean())
    lon_centre = float(data[x_dim].mean())
    lat_centre = float(data[y_dim].mean())

    dy = geod.Inverse(lat_centre, lon_centre, lat_centre + lat_spacing, lon_centre)[
        "s12"
    ]

    dx_rows = np.array(
        [
            geod.Inverse(float(lat), lon_centre, float(lat), lon_centre + lon_spacing)[
                "s12"
            ]
            for lat in data[y_dim].values
        ]
    )

    gy = np.gradient(z, dy, axis=0)
    gx = np.empty_like(z, dtype=float)
    for i, dx in enumerate(dx_rows):
        gx[i, :] = np.gradient(z[i, :], dx)

    return gy, gx, dy, dx_rows


# ============================================================================
# Public functions
# ============================================================================


def summary(data: xr.DataArray) -> pl.DataFrame:
    """
    Generate summary statistics.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data

    Returns
    -------
    pl.DataFrame
        DataFrame with statistics (count, mean, std, min, 25%, 50%, 75%, max)
    """
    values = _clean_values(data)
    return pl.DataFrame(
        {
            "statistic": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
            "value": [
                float(len(values)),
                float(np.mean(values)),
                float(np.std(values)),
                float(np.min(values)),
                float(np.percentile(values, 25)),
                float(np.median(values)),
                float(np.percentile(values, 75)),
                float(np.max(values)),
            ],
        }
    )


def hypsometric_index(data: xr.DataArray) -> float:
    """
    Calculate the hypsometric index (HI).

    HI = (mean - min) / (max - min)

    Parameters
    ----------
    data : xr.DataArray
        Elevation data

    Returns
    -------
    float
        Hypsometric index in [0, 1], or NaN for flat surfaces

    Examples
    --------
    >>> hi = hypsometric_index(data)
    """
    values = _clean_values(data)
    if len(values) == 0:
        return np.nan
    h_mean = np.mean(values)
    h_min = np.min(values)
    h_max = np.max(values)
    if h_max == h_min:
        return np.nan
    return float((h_mean - h_min) / (h_max - h_min))


def hypsometric_curve(
    data: xr.DataArray,
    bins: int = 100,
    absolute: bool = False,
) -> pl.DataFrame:
    """
    Calculate the hypsometric curve.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data.
    bins : int, default 100
        Number of elevation bins.
    absolute : bool, default False
        If False, return normalised relative_area (0–1) and
        relative_elevation (0–1).  If True, return absolute
        ``depth`` (metres) and ``cumulative_area`` (m²).

    Returns
    -------
    pl.DataFrame
        When *absolute* is False: columns ``relative_area`` and
        ``relative_elevation``.
        When *absolute* is True: columns ``depth`` (bin centres in
        metres) and ``cumulative_area`` (m²).

    Examples
    --------
    >>> df = hypsometric_curve(data)
    >>> plt.plot(df["relative_area"], df["relative_elevation"])

    Absolute depth-area curve:

    >>> df = hypsometric_curve(data, absolute=True)
    >>> plt.plot(df["cumulative_area"], df["depth"])
    """
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")

    z = data.values.astype(float)
    cell_area = _cell_areas(data)

    valid = ~np.isnan(z)
    z_valid = z[valid]
    area_valid = cell_area[valid]

    if len(z_valid) == 0:
        if absolute:
            return pl.DataFrame(
                schema={
                    "depth": pl.Float64,
                    "cumulative_area": pl.Float64,
                }
            )
        return pl.DataFrame(
            schema={
                "relative_area": pl.Float64,
                "relative_elevation": pl.Float64,
            }
        )

    h_min, h_max = z_valid.min(), z_valid.max()

    if h_max == h_min:
        if absolute:
            return pl.DataFrame(
                {
                    "depth": np.full(bins, h_min),
                    "cumulative_area": np.full(bins, float(area_valid.sum())),
                }
            )
        return pl.DataFrame(
            {
                "relative_area": np.ones(bins),
                "relative_elevation": np.linspace(0, 1, bins),
            }
        )

    bin_edges = np.linspace(h_min, h_max, bins + 1)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Sum cell area per bin
    indices = np.digitize(z_valid, bin_edges)
    indices = np.clip(indices, 1, bins)
    bin_area = np.zeros(bins)
    for i in range(len(z_valid)):
        bin_area[indices[i] - 1] += area_valid[i]

    # Cumulative from shallowest (highest elevation) down
    cumulative = np.cumsum(bin_area[::-1])[::-1]

    if absolute:
        return pl.DataFrame(
            {
                "depth": bin_centres,
                "cumulative_area": cumulative,
            }
        )

    return pl.DataFrame(
        {
            "relative_area": cumulative / cumulative[0],
            "relative_elevation": (bin_centres - h_min) / (h_max - h_min),
        }
    )


def slope(data: xr.DataArray) -> xr.DataArray:
    """
    Calculate seafloor slope in degrees.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data

    Returns
    -------
    xr.DataArray
        Slope magnitude in degrees
    """
    gy, gx, _, _ = _gradients(data)
    slope_deg = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    return xr.DataArray(slope_deg, coords=data.coords, dims=data.dims, name="slope")


def curvature(data: xr.DataArray) -> xr.DataArray:
    """
    Calculate seafloor curvature (Laplacian).

    Parameters
    ----------
    data : xr.DataArray
        Elevation data

    Returns
    -------
    xr.DataArray
        Curvature values (positive = convex/ridges, negative = concave/valleys)
    """
    gy, gx, dy, dx_rows = _gradients(data)
    gyy = np.gradient(gy, dy, axis=0)

    gxx = np.empty_like(gx, dtype=float)
    for i, dx in enumerate(dx_rows):
        gxx[i, :] = np.gradient(gx[i, :], dx)

    return xr.DataArray(gxx + gyy, coords=data.coords, dims=data.dims, name="curvature")


def bpi(data: xr.DataArray, radius_km: float = 1.0) -> xr.DataArray:
    """
    Calculate Bathymetric Position Index (BPI).

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    radius_km : float, default 1.0
        Radius of the neighbourhood in kilometres (square window)

    Returns
    -------
    xr.DataArray
        BPI values (positive = ridges, negative = valleys)

    Examples
    --------
    >>> bpi_data = bpi(data, radius_km=2.0)
    """
    if radius_km <= 0:
        raise ValueError(f"radius_km must be positive, got {radius_km}")

    from scipy.ndimage import uniform_filter  # noqa: PLC0415

    dy, dx = _cell_size_metres(data)
    cell_size = (dy + dx) / 2
    window_size = max(3, int(2 * radius_km * 1000 / cell_size) + 1)

    neighbourhood_mean = uniform_filter(
        data.values.astype(float), size=window_size, mode="nearest"
    )
    bpi_values = data.values - neighbourhood_mean

    return xr.DataArray(bpi_values, coords=data.coords, dims=data.dims, name="bpi")


def rugosity(data: xr.DataArray, radius_km: float = 1.0) -> xr.DataArray:
    """
    Calculate Vector Ruggedness Measure (VRM).

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    radius_km : float, default 1.0
        Radius of the neighbourhood in kilometres

    Returns
    -------
    xr.DataArray
        VRM values in range [0, 1]

    References
    ----------
    Sappington, J.M., Longshore, K.M., and Thompson, D.B. (2007).
    Quantifying landscape ruggedness for animal habitat analysis: a case
    study using bighorn sheep in the Mojave Desert. Journal of Wildlife
    Management, 71(5), 1419–1426.

    Examples
    --------
    >>> rug = rugosity(data, radius_km=0.5)
    """
    if radius_km <= 0:
        raise ValueError(f"radius_km must be positive, got {radius_km}")

    from scipy.ndimage import uniform_filter  # noqa: PLC0415

    gy, gx, dy, dx_rows = _gradients(data)

    slp = np.arctan(np.sqrt(gx**2 + gy**2))
    asp = np.arctan2(gy, gx)

    x = np.sin(slp) * np.sin(asp)
    y = np.sin(slp) * np.cos(asp)
    z = np.cos(slp)

    cell_size = (dy + float(dx_rows.mean())) / 2
    window_size = max(3, int(2 * radius_km * 1000 / cell_size) + 1)

    x_mean = uniform_filter(x, size=window_size, mode="nearest")
    y_mean = uniform_filter(y, size=window_size, mode="nearest")
    z_mean = uniform_filter(z, size=window_size, mode="nearest")

    vrm = 1.0 - np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)

    return xr.DataArray(vrm, coords=data.coords, dims=data.dims, name="rugosity")


def aspect(data: xr.DataArray) -> xr.DataArray:
    """
    Calculate seafloor aspect (downslope direction).

    Aspect is the compass direction a slope faces, measured in degrees
    clockwise from north. This follows the standard GIS convention
    (ESRI, GDAL, GRASS). Flat areas are returned as NaN.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data

    Returns
    -------
    xr.DataArray
        Aspect in degrees [0, 360), NaN where slope is zero

    Examples
    --------
    >>> asp = aspect(data)
    """
    gy, gx, _, _ = _gradients(data)

    asp = (270 - np.degrees(np.arctan2(gy, gx))) % 360
    asp[np.sqrt(gx**2 + gy**2) < 1e-10] = np.nan

    return xr.DataArray(asp, coords=data.coords, dims=data.dims, name="aspect")


def geomorphons(
    data: xr.DataArray,
    lookup_km: float = 2.0,
    flatness_threshold: float = 1.0,
) -> xr.DataArray:
    """
    Classify seafloor morphology using geomorphons.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    lookup_km : float, default 2.0
        Lookup distance in kilometres.
    flatness_threshold : float, default 1.0
        Angle threshold in degrees below which differences are treated as flat.

    Returns
    -------
    xr.DataArray
        Integer class codes (1–10):
        1=flat, 2=peak, 3=ridge, 4=shoulder, 5=spur,
        6=slope, 7=hollow, 8=footslope, 9=valley, 10=pit

    Notes
    -----
    Each of the eight cardinal/diagonal directions is scanned from 1 cell
    to ``lookup_km``, recording the maximum and minimum elevation angles
    along the full line of sight.  A direction is classified as *higher*
    when the maximum angle exceeds ``flatness_threshold`` and *lower*
    when the minimum angle falls below it.

    Cells within ``lookup_km`` of the grid edge receive fewer directional
    comparisons and may be misclassified. Consider trimming edges for
    critical analyses.

    References
    ----------
    Jasiewicz, J., & Stepinski, T.F. (2013). Geomorphons — a pattern
    recognition approach to classification and mapping of landforms.
    Geomorphology, 182, 147–156.

    Examples
    --------
    >>> geom = geomorphons(data, lookup_km=2.0)
    """
    if lookup_km <= 0:
        raise ValueError(f"lookup_km must be positive, got {lookup_km}")
    if flatness_threshold <= 0:
        raise ValueError(
            f"flatness_threshold must be positive, got {flatness_threshold}"
        )

    dy, dx = _cell_size_metres(data)

    z = data.values.astype(float)
    ny, nx = z.shape

    directions = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ]

    threshold_rad = np.radians(flatness_threshold)
    p = np.zeros((ny, nx), dtype=np.int8)
    m = np.zeros((ny, nx), dtype=np.int8)

    for drow, dcol in directions:
        step_dist = np.sqrt((drow * dy) ** 2 + (dcol * dx) ** 2)
        n_steps = max(1, round(lookup_km * 1000 / step_dist))

        max_angle = np.full((ny, nx), -np.inf)
        min_angle = np.full((ny, nx), np.inf)

        for step in range(1, n_steps + 1):
            horiz = step * step_dist
            row_off = drow * step
            col_off = dcol * step
            i0 = max(0, -row_off)
            i1 = ny - max(0, row_off)
            j0 = max(0, -col_off)
            j1 = nx - max(0, col_off)

            if i0 >= i1 or j0 >= j1:
                continue

            dz = (
                z[i0 + row_off : i1 + row_off, j0 + col_off : j1 + col_off]
                - z[i0:i1, j0:j1]
            )
            angle = np.arctan2(dz, horiz)
            max_angle[i0:i1, j0:j1] = np.maximum(max_angle[i0:i1, j0:j1], angle)
            min_angle[i0:i1, j0:j1] = np.minimum(min_angle[i0:i1, j0:j1], angle)

        p += (max_angle > threshold_rad).astype(np.int8)
        m += (min_angle < -threshold_rad).astype(np.int8)

    geom = np.full((ny, nx), 6, dtype=np.int8)
    geom[(p == 0) & (m == 0)] = 1
    geom[(p == 0) & (m >= 5)] = 2
    geom[(p == 0) & (0 < m) & (m < 5)] = 3
    geom[(m == 0) & (0 < p) & (p < 5)] = 9
    geom[(m == 0) & (p >= 5)] = 10
    geom[(m > p) & (p > 0) & ((m - p) >= 3)] = 4
    geom[(m > p) & (p > 0) & ((m - p) < 3)] = 5
    geom[(p > m) & (m > 0) & ((p - m) >= 3)] = 8
    geom[(p > m) & (m > 0) & ((p - m) < 3)] = 7

    return xr.DataArray(geom, coords=data.coords, dims=data.dims, name="geomorphons")


def contours(
    data: xr.DataArray,
    levels: np.ndarray | Sequence[float] | None = None,
    interval: float | None = None,
) -> gpd.GeoDataFrame:
    """
    Extract depth contour lines as vector geometries.

    Provide *levels* for explicit contour values, *interval* for regular
    spacing, or neither to let matplotlib choose automatically.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data.
    levels : array-like, optional
        Explicit contour elevations (e.g. ``[-100, -200, -500]``).
    interval : float, optional
        Regular contour spacing in metres.  Ignored when *levels* is given.

    Returns
    -------
    gpd.GeoDataFrame
        Columns ``depth`` and ``geometry`` (LineString).

    Raises
    ------
    ValueError
        If *interval* is not positive.

    Examples
    --------
    >>> gdf = contours(data, interval=200)
    """
    if interval is not None and interval <= 0:
        raise ValueError(f"interval must be positive, got {interval}")

    x_dim, y_dim = get_dim_names(data)
    x = data[x_dim].values
    y = data[y_dim].values
    z = data.values.astype(float)

    if levels is not None:
        levels = np.sort(levels)
    elif interval is not None:
        values = z[~np.isnan(z)]
        lo = np.floor(values.min() / interval) * interval
        hi = np.ceil(values.max() / interval) * interval
        levels = np.arange(lo, hi + interval / 2, interval)

    fig, ax = plt.subplots()
    try:
        cs = (
            ax.contour(x, y, z)
            if levels is None
            else ax.contour(x, y, z, levels=levels)
        )
        rows: list[dict] = []
        for level_value, segments in zip(cs.levels, cs.allsegs, strict=False):
            for seg in segments:
                if len(seg) >= 2:
                    rows.append(
                        {"depth": float(level_value), "geometry": LineString(seg)}
                    )
    finally:
        plt.close(fig)

    return gpd.GeoDataFrame(rows, crs=get_crs(data))


def smooth(
    data: xr.DataArray,
    sigma_km: float = 1.0,
) -> xr.DataArray:
    """
    Gaussian smooth a bathymetry grid.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data.
    sigma_km : float, default 1.0
        Gaussian kernel standard deviation in kilometres.

    Returns
    -------
    xr.DataArray
        Smoothed elevation grid (NaNs propagated).

    Examples
    --------
    >>> smoothed = smooth(data, sigma_km=2.0)
    """
    if sigma_km <= 0:
        raise ValueError(f"sigma_km must be positive, got {sigma_km}")

    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    dy, dx = _cell_size_metres(data)
    cell_size = (dy + dx) / 2
    sigma_pixels = sigma_km * 1000 / cell_size

    values = data.values.astype(float)
    mask = np.isnan(values)

    # Normalised convolution: smooth ignoring NaNs, then restore the mask
    filled = np.where(mask, 0.0, values)
    weights = np.where(mask, 0.0, 1.0)

    smoothed = gaussian_filter(filled, sigma=sigma_pixels, mode="nearest")
    weight_sum = gaussian_filter(weights, sigma=sigma_pixels, mode="nearest")

    with np.errstate(invalid="ignore"):
        result = np.where(weight_sum > 0, smoothed / weight_sum, np.nan)
    result[mask] = np.nan

    return xr.DataArray(result, coords=data.coords, dims=data.dims, name="elevation")


# ============================================================================
# Volume & area calculations
# ============================================================================


def _cell_areas(data: xr.DataArray) -> np.ndarray:
    """Return a 2-D array of per-cell planimetric areas in m².

    For projected CRS every cell has the same area.  For geographic CRS
    the east-west extent shrinks with latitude, so areas are computed
    per row using the same geodesic logic as ``_cell_size_metres``.
    """
    dy, dx = _cell_size_metres(data)

    if is_projected(data):
        return np.full(data.shape, dy * dx)

    # Geographic: dx varies by latitude row
    x_dim, y_dim = get_dim_names(data)
    geod = Geodesic.WGS84
    lon_spacing = np.abs(np.diff(data[x_dim].values).mean())
    lon_centre = float(data[x_dim].mean())

    areas = np.empty(data.shape, dtype=float)
    for i, lat in enumerate(data[y_dim].values):
        row_dx = geod.Inverse(
            float(lat), lon_centre, float(lat), lon_centre + lon_spacing
        )["s12"]
        areas[i, :] = dy * row_dx

    return areas


def volume(
    data: xr.DataArray,
    upper_level: float = 0,
    lower_level: float | None = None,
) -> float:
    """
    Calculate water volume between two depth levels.

    Volume is computed as the sum of per-cell water columns multiplied
    by the planimetric cell area.  Only cells whose elevation falls
    between *lower_level* and *upper_level* contribute.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data (negative = below sea level).
    upper_level : float, default 0
        Upper bounding elevation in metres.
    lower_level : float, optional
        Lower bounding elevation in metres.  Defaults to the minimum
        elevation in *data*.

    Returns
    -------
    float
        Volume in m³.

    Examples
    --------
    >>> vol = volume(data, upper_level=0, lower_level=-500)
    """
    z = data.values.astype(float)

    if lower_level is None:
        lower_level = float(np.nanmin(z))

    if upper_level < lower_level:
        raise ValueError(
            f"upper_level ({upper_level}) must be >= lower_level ({lower_level})"
        )

    cell_area = _cell_areas(data)

    # Water column: clamp elevation into [lower, upper], measure gap to upper
    water_col = upper_level - np.clip(z, lower_level, upper_level)
    water_col = np.where(np.isnan(z), 0.0, water_col)

    return float(np.sum(water_col * cell_area))


def area(
    data: xr.DataArray,
    upper_level: float = 0,
    lower_level: float | None = None,
    true_surface: bool = False,
) -> float:
    """
    Calculate seafloor area between two depth levels.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data (negative = below sea level).
    upper_level : float, default 0
        Upper bounding elevation in metres.
    lower_level : float, optional
        Lower bounding elevation in metres.  Defaults to the minimum
        elevation in *data*.
    true_surface : bool, default False
        If True, compute true (slope-corrected) surface area rather than
        planimetric area.  Each cell's area is divided by cos(slope).

    Returns
    -------
    float
        Area in m².

    Examples
    --------
    >>> a = area(data, upper_level=0, lower_level=-200)
    """
    z = data.values.astype(float)

    if lower_level is None:
        lower_level = float(np.nanmin(z))

    if upper_level < lower_level:
        raise ValueError(
            f"upper_level ({upper_level}) must be >= lower_level ({lower_level})"
        )

    cell_area = _cell_areas(data)
    mask = ~np.isnan(z) & (z <= upper_level) & (z >= lower_level)

    if true_surface:
        gy, gx, _, _ = _gradients(data)
        slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
        with np.errstate(divide="ignore", invalid="ignore"):
            surface_factor = 1.0 / np.cos(slope_rad)
        surface_factor = np.where(np.isfinite(surface_factor), surface_factor, 1.0)
        cell_area = cell_area * surface_factor

    return float(np.sum(np.where(mask, cell_area, 0.0)))
