"""Grid operations: clip, resample, reproject, merge, and gap filling."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import CRS
from rasterio.enums import Resampling
from scipy.interpolate import griddata
from shapely.geometry import box

from bathy.io import REGIONS
from bathy.utils import get_crs, get_dim_names, is_projected

_RESAMPLING_METHODS = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
    "lanczos": Resampling.lanczos,
    "average": Resampling.average,
    "min": Resampling.min,
    "max": Resampling.max,
}


# ============================================================================
# Internal helpers
# ============================================================================


def _normalize_dims(data: xr.DataArray) -> xr.DataArray:
    """Rename dims to match bathy convention based on CRS.

    Geographic CRS (or no CRS) → ``lon``/``lat``.
    Projected CRS → ``x``/``y``.
    """
    crs = get_crs(data)
    dims = set(data.dims)

    if crs is not None and crs.is_projected:
        if {"lon", "lat"} <= dims:
            data = data.rename({"lon": "x", "lat": "y"})
    else:
        if {"x", "y"} <= dims:
            data = data.rename({"x": "lon", "y": "lat"})

    return data


def _prepare_spatial(data: xr.DataArray) -> xr.DataArray:
    """Ensure *data* has a CRS and spatial dims set for rioxarray ops."""
    if get_crs(data) is None:
        data = data.rio.write_crs("EPSG:4326")
    x_dim, y_dim = get_dim_names(data)
    return data.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)


def _get_resampling(method: str) -> Resampling:
    """Look up a rasterio ``Resampling`` enum by name."""
    if method not in _RESAMPLING_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {sorted(_RESAMPLING_METHODS)}"
        )
    return _RESAMPLING_METHODS[method]


def _metres_to_resolution(data: xr.DataArray, metres: float) -> float:
    """Convert a distance in metres to CRS-native resolution.

    For projected CRS, returns *metres* unchanged.  For geographic CRS,
    converts to degrees using the grid's centre latitude.
    """
    if is_projected(data):
        return metres

    _, y_dim = get_dim_names(data)
    lat_centre = float(data[y_dim].values.mean())
    # 1 degree latitude ≈ 111 320 m; longitude shrinks by cos(lat)
    deg_lat = metres / 111_320
    deg_lon = metres / (111_320 * np.cos(np.radians(lat_centre)))
    # Use the average as a single resolution value
    return (deg_lat + deg_lon) / 2


def _region_to_geodataframe(region: str) -> gpd.GeoDataFrame:
    """Convert a named region preset to a GeoDataFrame with a box geometry."""
    name = region.lower().replace(" ", "_")
    if name not in REGIONS:
        available = ", ".join(sorted(REGIONS))
        raise ValueError(f"Unknown region '{region}'. Available: {available}")
    lon_min, lon_max, lat_min, lat_max = REGIONS[name]
    return gpd.GeoDataFrame(
        geometry=[box(lon_min, lat_min, lon_max, lat_max)], crs="EPSG:4326"
    )


# ============================================================================
# Public functions
# ============================================================================


def clip(
    data: xr.DataArray,
    geometry: gpd.GeoDataFrame | str | Path | None = None,
    region: str | None = None,
) -> xr.DataArray:
    """Clip a grid to a polygon boundary or region preset.

    Parameters
    ----------
    data : xr.DataArray
        Elevation grid.
    geometry : GeoDataFrame, str, or Path, optional
        Clipping geometry. Can be a GeoDataFrame, or a path to a vector
        file (Shapefile, GeoPackage, GeoJSON, etc.).
    region : str, optional
        Named region preset (see :func:`bathy.list_regions`).
        Cannot be used together with *geometry*.

    Returns
    -------
    xr.DataArray
        Clipped grid with dims normalised to bathy conventions.

    Raises
    ------
    ValueError
        If both or neither of *geometry* and *region* are specified.

    Examples
    --------
    Clip to a region preset:

    >>> clipped = bathy.clip(data, region="mediterranean")

    Clip to a custom polygon:

    >>> from shapely.geometry import box
    >>> gdf = gpd.GeoDataFrame(geometry=[box(-8, 51, -7, 53)], crs="EPSG:4326")
    >>> clipped = bathy.clip(data, geometry=gdf)
    """
    if (geometry is None) == (region is None):
        raise ValueError("Specify exactly one of 'geometry' or 'region'.")

    if region is not None:
        gdf = _region_to_geodataframe(region)
    elif isinstance(geometry, (str, Path)):
        gdf = gpd.read_file(geometry)
    else:
        # geometry is guaranteed non-None by the check above
        gdf: gpd.GeoDataFrame = geometry  # ty: ignore[invalid-assignment]

    data = _prepare_spatial(data)

    # Reproject clipping geometry to match data CRS if needed
    if gdf.crs is not None and not gdf.crs.equals(data.rio.crs):
        gdf = gdf.to_crs(data.rio.crs)

    clipped = data.rio.clip(gdf.geometry, drop=True)
    return _normalize_dims(clipped)


def resample(
    data: xr.DataArray,
    resolution_degrees: float | None = None,
    resolution_m: float | None = None,
    method: str = "bilinear",
) -> xr.DataArray:
    """Resample a grid to a new resolution.

    Parameters
    ----------
    data : xr.DataArray
        Elevation grid.
    resolution_degrees : float, optional
        Target resolution in degrees. For projected CRS this is
        converted to metres. Exactly one of *resolution_degrees* or
        *resolution_m* must be given.
    resolution_m : float, optional
        Target resolution in metres. For geographic CRS this is
        converted to degrees using the grid's centre latitude.
    method : str, optional
        Resampling method. One of ``'nearest'``, ``'bilinear'``,
        ``'cubic'``, ``'cubic_spline'``, ``'lanczos'``, ``'average'``,
        ``'min'``, ``'max'``. Default ``'bilinear'``.

    Returns
    -------
    xr.DataArray
        Resampled grid with dims normalised to bathy conventions.

    Examples
    --------
    Resample to approximately 500 m:

    >>> resampled = bathy.resample(data, resolution_m=500)

    Resample to 0.01°:

    >>> coarser = bathy.resample(data, resolution_degrees=0.01)
    """
    if (resolution_degrees is None) == (resolution_m is None):
        raise ValueError(
            "Specify exactly one of 'resolution_degrees' or 'resolution_m'."
        )

    if resolution_m is not None:
        resolution = _metres_to_resolution(data, resolution_m)
    else:
        resolution = resolution_degrees

    data = _prepare_spatial(data)
    resampled = data.rio.reproject(
        data.rio.crs,
        resolution=resolution,
        resampling=_get_resampling(method),
    )
    return _normalize_dims(resampled)


def reproject(
    data: xr.DataArray,
    target_crs: str | int | CRS,
    resolution: float | None = None,
    method: str = "bilinear",
) -> xr.DataArray:
    """Reproject a grid to a different coordinate reference system.

    Dim names are automatically updated: ``lon``/``lat`` for geographic
    CRS, ``x``/``y`` for projected CRS.

    Parameters
    ----------
    data : xr.DataArray
        Elevation grid.
    target_crs : str, int, or pyproj.CRS
        Target CRS (e.g. ``"EPSG:32629"``).
    resolution : float, optional
        Target resolution in the units of *target_crs*. If ``None``,
        rioxarray picks an appropriate resolution automatically.
    method : str, optional
        Resampling method (default ``'bilinear'``). See :func:`resample`.

    Returns
    -------
    xr.DataArray
        Reprojected grid with dim names matching bathy conventions.

    Examples
    --------
    Reproject to UTM zone 29N:

    >>> utm = bathy.reproject(data, target_crs="EPSG:32629")
    >>> utm.dims  # ('y', 'x') — projected convention
    """
    data = _prepare_spatial(data)

    kwargs: dict = {"resampling": _get_resampling(method)}
    if resolution is not None:
        kwargs["resolution"] = resolution

    reprojected = data.rio.reproject(target_crs, **kwargs)
    return _normalize_dims(reprojected)


def merge(
    datasets: list[xr.DataArray],
    method: str = "mean",
    resolution: float | None = None,
) -> xr.DataArray:
    """Merge multiple grids into a single grid.

    Overlapping cells are combined using *method*. When grids have
    different resolutions they are resampled to the finest resolution
    (or to *resolution* if given) before merging.

    Parameters
    ----------
    datasets : list of xr.DataArray
        Grids to merge. Must share the same CRS.
    method : str, optional
        How to combine overlapping cells: ``'mean'``, ``'min'``,
        ``'max'``, or ``'first'``. Default ``'mean'``.
    resolution : float, optional
        Target resolution. If ``None``, the finest input resolution is
        used.

    Returns
    -------
    xr.DataArray
        Merged grid with dims normalised to bathy conventions.

    Raises
    ------
    ValueError
        If fewer than two datasets are given, CRS do not match, or
        *method* is unknown.

    Examples
    --------
    Merge two adjacent tiles, averaging overlapping cells:

    >>> merged = bathy.merge([tile_west, tile_east])

    Merge keeping the minimum depth in overlaps:

    >>> merged = bathy.merge([gebco, emodnet], method="min")
    """
    if len(datasets) < 2:
        raise ValueError("Need at least 2 datasets to merge.")

    valid_methods = {"mean", "min", "max", "first"}
    if method not in valid_methods:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {sorted(valid_methods)}"
        )

    # Determine common CRS
    crs_list = [get_crs(d) for d in datasets]
    ref_crs = crs_list[0] or CRS.from_epsg(4326)
    for i, c in enumerate(crs_list[1:], 1):
        effective = c or CRS.from_epsg(4326)
        if not effective.equals(ref_crs):
            raise ValueError(
                f"CRS mismatch: dataset 0 has {ref_crs}, dataset {i} has "
                f"{effective}. Reproject grids to a common CRS first."
            )

    # Determine target resolution (finest among inputs)
    if resolution is None:
        resolutions = []
        for d in datasets:
            x_dim, _ = get_dim_names(d)
            resolutions.append(float(np.abs(np.diff(d[x_dim].values).mean())))
        resolution = min(resolutions)

    # Resample all datasets to common resolution
    use_projected = is_projected(datasets[0])
    aligned = []
    for d in datasets:
        d = _prepare_spatial(d)
        x_dim, _ = get_dim_names(d)
        current_res = float(np.abs(np.diff(d[x_dim].values).mean()))
        if not np.isclose(current_res, resolution, rtol=0.01):
            if use_projected:
                d = resample(d, resolution_m=resolution)
            else:
                d = resample(d, resolution_degrees=resolution)
        aligned.append(d)

    # Build common coordinate grid spanning all inputs
    x_dim, y_dim = get_dim_names(aligned[0])
    all_x = np.concatenate([d[x_dim].values for d in aligned])
    all_y = np.concatenate([d[y_dim].values for d in aligned])
    new_x = np.arange(
        float(all_x.min()), float(all_x.max()) + resolution / 2, resolution
    )
    new_y = np.arange(
        float(all_y.min()), float(all_y.max()) + resolution / 2, resolution
    )

    # Reindex each dataset onto the common grid, then combine
    reindexed = []
    for d in aligned:
        d_x, d_y = get_dim_names(d)
        d = d.reindex(
            {d_x: new_x, d_y: new_y}, method="nearest", tolerance=resolution * 0.6
        )
        reindexed.append(d)

    stacked = xr.concat(reindexed, dim="__source__")

    if method == "mean":
        result = stacked.mean(dim="__source__")
    elif method == "min":
        result = stacked.min(dim="__source__")
    elif method == "max":
        result = stacked.max(dim="__source__")
    else:  # first
        result = stacked.isel(__source__=0)
        for i in range(1, len(reindexed)):
            mask = result.isnull()
            result = result.where(~mask, stacked.isel(__source__=i))

    result.name = "elevation"
    if ref_crs is not None:
        result = result.rio.write_crs(ref_crs)
    return _normalize_dims(result)


def fill_gaps(
    data: xr.DataArray,
    method: str = "nearest",
) -> xr.DataArray:
    """Fill NaN gaps in a grid using interpolation.

    Parameters
    ----------
    data : xr.DataArray
        Elevation grid with NaN gaps.
    method : str, optional
        Interpolation method: ``'nearest'`` or ``'linear'``.
        Default ``'nearest'``.

    Returns
    -------
    xr.DataArray
        Grid with NaN values filled.

    Raises
    ------
    ValueError
        If *method* is not recognised.

    Examples
    --------
    >>> filled = bathy.fill_gaps(data, method="linear")
    """
    valid = {"nearest", "linear"}
    if method not in valid:
        raise ValueError(f"Unknown method '{method}'. Choose from: {sorted(valid)}")

    x_dim, y_dim = get_dim_names(data)
    z = data.values.astype(float)
    mask = np.isnan(z)

    if not mask.any():
        return data

    xs = data[x_dim].values
    ys = data[y_dim].values
    xx, yy = np.meshgrid(xs, ys)

    known = ~mask
    filled = griddata(
        (xx[known], yy[known]),
        z[known],
        (xx, yy),
        method=method,
    )

    return data.copy(data=filled)
