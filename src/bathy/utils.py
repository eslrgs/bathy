"""CRS detection and coordinate utilities."""

import xarray as xr
from pyproj import CRS


def get_crs(data: xr.DataArray) -> CRS | None:
    """Return the CRS attached to *data*, or ``None`` if absent.

    Uses the rioxarray ``.rio.crs`` accessor.  Returns ``None`` when
    rioxarray is not imported or no CRS metadata exists.
    """
    try:
        crs = data.rio.crs
    except AttributeError:
        return None
    return CRS(crs) if crs is not None else None


def is_geographic(data: xr.DataArray) -> bool:
    """Return ``True`` if *data* has a geographic CRS (or no CRS)."""
    crs = get_crs(data)
    return crs is None or crs.is_geographic


def is_projected(data: xr.DataArray) -> bool:
    """Return ``True`` if *data* has a projected CRS."""
    crs = get_crs(data)
    return crs is not None and crs.is_projected


def get_dim_names(data: xr.DataArray) -> tuple[str, str]:
    """Return ``(x_dim, y_dim)`` dimension names present in *data*.

    Returns ``("lon", "lat")`` for geographic data and ``("x", "y")``
    for projected data.

    Raises
    ------
    ValueError
        If neither ``lon``/``lat`` nor ``x``/``y`` dimensions are found.
    """
    dims = set(data.dims)
    if {"lon", "lat"} <= dims:
        return "lon", "lat"
    if {"x", "y"} <= dims:
        return "x", "y"
    raise ValueError(
        f"Expected dimensions ('lon', 'lat') or ('x', 'y'), got {sorted(dims)}"
    )


def axis_labels(data: xr.DataArray) -> tuple[str, str]:
    """Return ``(x_label, y_label)`` for plot axes."""
    if is_projected(data):
        return "Easting (m)", "Northing (m)"
    return "Longitude (°)", "Latitude (°)"


def crs_axis_labels(crs: CRS | None) -> tuple[str, str]:
    """Return ``(x_label, y_label)`` from a CRS object (or ``None``)."""
    if crs is not None and crs.is_projected:
        return "Easting (m)", "Northing (m)"
    return "Longitude (°)", "Latitude (°)"
