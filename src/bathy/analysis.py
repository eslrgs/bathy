"""Bathymetry analysis functions."""

import numpy as np
import polars as pl
import xarray as xr
from geographiclib.geodesic import Geodesic

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
    from bathy.utils import get_dim_names, is_projected  # noqa: PLC0415

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
    from bathy.utils import get_dim_names, is_projected  # noqa: PLC0415

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
    data: xr.DataArray, bins: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the hypsometric curve.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    bins : int, default 100
        Number of elevation bins

    Returns
    -------
    relative_area : np.ndarray
        Cumulative proportion of area above each elevation (1 to 0)
    relative_elevation : np.ndarray
        Normalised elevation (0 = min, 1 = max)

    Examples
    --------
    >>> rel_area, rel_elev = hypsometric_curve(data)
    >>> plt.plot(rel_area, rel_elev)
    """
    values = _clean_values(data)
    if len(values) == 0:
        return np.array([]), np.array([])
    h_min, h_max = values.min(), values.max()
    if h_max == h_min:
        return np.ones(bins), np.linspace(0, 1, bins)

    bin_edges = np.linspace(h_min, h_max, bins + 1)
    counts, _ = np.histogram(values, bins=bin_edges)

    cumulative = np.cumsum(counts[::-1])[::-1]
    relative_area = cumulative / cumulative[0]

    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    relative_elevation = (bin_centres - h_min) / (h_max - h_min)

    return relative_area, relative_elevation


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
