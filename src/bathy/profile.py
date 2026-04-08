"""Bathymetric profile functions."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import polars as pl
import xarray as xr
from geographiclib.geodesic import Geodesic
from pyproj import CRS
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from shapely.geometry import LineString

if TYPE_CHECKING:
    from xarray.core.types import InterpOptions

from bathy.utils import get_crs, get_dim_names

logger = logging.getLogger(__name__)

# Default prominence threshold as fraction of total relief
_DEFAULT_PROMINENCE_FRACTION = 0.1


@dataclass(eq=False)
class Profile:
    """
    Single bathymetric profile.

    Attributes
    ----------
    distances : np.ndarray
        Distances along profile (m)
    elevations : np.ndarray
        Elevation values along profile (m)
    start_x, start_y : float
        Starting coordinates (lon/lat or easting/northing)
    end_x, end_y : float
        Ending coordinates (lon/lat or easting/northing)
    name : str, optional
        Profile name
    crs : pyproj.CRS, optional
        Coordinate reference system
    metadata : dict
        Additional metadata (includes path_xs, path_ys for plotting)

    Examples
    --------
    >>> prof = extract_profile(data, start=(-9.5, 52.0), end=(-5.5, 52.0))
    >>> prof = extract_profile(
    ...     data, start=(-9.5, 52.0), end=(-5.5, 52.0), point_spacing=1000.0
    ... )
    """

    distances: np.ndarray
    elevations: np.ndarray
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    name: str | None = None
    crs: CRS | None = None
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        name = f'"{self.name}", ' if self.name else ""
        dist_km = self.distances[-1] / 1000
        max_depth = np.nanmin(self.elevations)
        return f"Profile({name}{dist_km:.1f} km, max_depth={max_depth:.0f} m)"


# ============================================================================
# Internal helpers
# ============================================================================


def _validate_coordinates(
    data: xr.DataArray, x: float, y: float, param_name: str
) -> None:
    """Validate that coordinates are within data bounds."""
    x_dim, y_dim = get_dim_names(data)
    x_min, x_max = float(data[x_dim].min()), float(data[x_dim].max())
    y_min, y_max = float(data[y_dim].min()), float(data[y_dim].max())

    if not (x_min <= x <= x_max):
        raise ValueError(
            f"{param_name} {x_dim} ({x}) is outside "
            f"DEM bounds [{x_min:.2f}, {x_max:.2f}]"
        )
    if not (y_min <= y <= y_max):
        raise ValueError(
            f"{param_name} {y_dim} ({y}) is outside "
            f"DEM bounds [{y_min:.2f}, {y_max:.2f}]"
        )


def _ensure_descending(
    distances: np.ndarray, elevations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure profile descends from higher to lower elevation values.

    Parameters
    ----------
    distances : np.ndarray
        Distance values along profile
    elevations : np.ndarray
        Elevation values along profile

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (distances, elevations) — flipped and re-zeroed if start elevation
        < end elevation
    """
    if elevations[0] < elevations[-1]:
        new_dist = distances[-1] - distances[::-1]
        return new_dist, elevations[::-1]
    return distances, elevations


def _calculate_num_points(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    num_points: int | None,
    point_spacing: float | None,
    crs: CRS | None = None,
) -> int:
    """Calculate number of points for the profile."""
    if num_points is None and point_spacing is None:
        return 100
    if num_points is not None and point_spacing is not None:
        raise ValueError(
            "Cannot specify both num_points and point_spacing. Choose one."
        )

    if num_points is not None:
        if num_points < 1:
            raise ValueError(f"num_points must be at least 1, got {num_points}")
        return num_points

    assert point_spacing is not None  # noqa: S101
    if point_spacing <= 0:
        raise ValueError(f"point_spacing must be positive, got {point_spacing}")

    if crs is not None and crs.is_projected:
        total_distance_m = math.hypot(end_x - start_x, end_y - start_y)
    else:
        geod = Geodesic.WGS84
        result = geod.Inverse(start_y, start_x, end_y, end_x)
        total_distance_m = result["s12"]
    return max(2, int(np.ceil(total_distance_m / point_spacing)) + 1)


_VALID_METHODS = {"nearest", "linear", "cubic", "quadratic", "slinear", "zero"}


def _sample_grid(
    data: xr.DataArray,
    coords: dict[str, xr.DataArray | float],
    method: str,
) -> np.ndarray:
    """Sample *data* at *coords* using the given interpolation method."""
    if method == "nearest":
        return data.sel(coords, method="nearest").values.astype(float)
    interp_method: InterpOptions = method  # ty: ignore[invalid-assignment]
    return data.interp(coords, method=interp_method).values.astype(float)


def _validate_method(method: str) -> None:
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method {method!r}, expected one of {sorted(_VALID_METHODS)}"
        )


def _extract_profile_arrays(
    data: xr.DataArray,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    n: int,
    crs: CRS | None = None,
    method: str = "nearest",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract elevation and distance arrays along a profile.

    For geographic CRS, points are placed along a geodesic.  For projected
    CRS, points are placed along a straight line with Euclidean distances.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (elevations, distances_m) arrays
    """
    x_dim, y_dim = get_dim_names(data)

    if crs is not None and crs.is_projected:
        xs = np.linspace(start_x, end_x, n)
        ys = np.linspace(start_y, end_y, n)
        distances_m = np.sqrt((xs - xs[0]) ** 2 + (ys - ys[0]) ** 2)
    else:
        geod = Geodesic.WGS84
        line = geod.InverseLine(start_y, start_x, end_y, end_x)
        total_m = line.s13
        distances_m = np.linspace(0, total_m, n)

        xs = np.zeros(n)
        ys = np.zeros(n)
        for i, d_m in enumerate(distances_m):
            pos = line.Position(d_m)
            xs[i] = pos["lon2"]
            ys[i] = pos["lat2"]

    x_da = xr.DataArray(xs, dims="points")
    y_da = xr.DataArray(ys, dims="points")
    elevations = _sample_grid(data, {x_dim: x_da, y_dim: y_da}, method)

    return elevations, distances_m


def _interp_crossing(
    elevations: np.ndarray,
    distances: np.ndarray,
    target_elev: float,
) -> float | None:
    """Find the first distance where *elevations* crosses *target_elev*.

    The caller controls search direction by passing slices in the desired
    order (e.g. reversed for searching outward from a trough to the left).

    Returns ``None`` when no crossing exists.
    """
    diff = elevations - target_elev
    # Treat exact zeros as positive so np.sign doesn't produce 0.
    signs = np.sign(diff)
    signs[signs == 0] = 1
    change_idx = np.where(np.diff(signs))[0]
    if len(change_idx) == 0:
        return None
    i = change_idx[0]
    e0, e1 = elevations[i], elevations[i + 1]
    denom = e1 - e0
    if denom == 0:
        return float(distances[i])
    t = (target_elev - e0) / denom
    return float(distances[i] + t * (distances[i + 1] - distances[i]))


def _normalise_profile(
    distances: np.ndarray, elevations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalise distances and elevations to [0, 1].

    Parameters
    ----------
    distances : np.ndarray
        Distance values along profile
    elevations : np.ndarray
        Elevation values along profile

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (distances, elevations) normalised to [0, 1]
    """
    elev_min, elev_max = float(elevations.min()), float(elevations.max())
    if elev_max > elev_min:
        elevations = (elevations - elev_min) / (elev_max - elev_min)
    dist_min, dist_max = float(distances.min()), float(distances.max())
    if dist_max > dist_min:
        distances = (distances - dist_min) / (dist_max - dist_min)
    return distances, elevations


# ============================================================================
# Construction functions
# ============================================================================


def extract_profile(
    data: xr.DataArray,
    start: tuple[float, float],
    end: tuple[float, float],
    num_points: int | None = None,
    point_spacing: float | None = None,
    name: str | None = None,
    metadata: dict | None = None,
    method: str = "nearest",
) -> Profile:
    """
    Create a bathymetric profile between two points.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    start : tuple[float, float]
        Starting coordinates (lon, lat) or (easting, northing)
    end : tuple[float, float]
        Ending coordinates (lon, lat) or (easting, northing)
    num_points : int, optional
        Number of points along profile. Cannot be used with point_spacing.
        Default: 100 if neither num_points nor point_spacing is specified.
    point_spacing : float, optional
        Spacing between points in metres. Cannot be used with num_points.
    name : str, optional
        Name for this profile
    metadata : dict, optional
        Additional metadata
    method : str, optional
        Interpolation method for sampling the grid: ``"nearest"`` (default),
        ``"linear"``, ``"cubic"``, ``"quadratic"``, ``"slinear"``, or
        ``"zero"``.  All methods except ``"nearest"`` use
        :meth:`xarray.DataArray.interp` and may produce NaN at grid edges
        or where the input contains NaN.

    Returns
    -------
    Profile

    Examples
    --------
    >>> prof = extract_profile(data, start=(-9.5, 52.0), end=(-5.5, 52.0))
    >>> prof = extract_profile(
    ...     data, start=(-9.5, 52.0), end=(-5.5, 52.0), point_spacing=1000.0
    ... )
    """
    _validate_method(method)
    start_x, start_y = start
    end_x, end_y = end
    crs = get_crs(data)

    _validate_coordinates(data, start_x, start_y, "start")
    _validate_coordinates(data, end_x, end_y, "end")

    n = _calculate_num_points(
        start_x, start_y, end_x, end_y, num_points, point_spacing, crs
    )
    elevations, distances = _extract_profile_arrays(
        data, start_x, start_y, end_x, end_y, n, crs, method=method
    )

    meta = dict(metadata) if metadata else {}
    meta["path_xs"] = [start_x, end_x]
    meta["path_ys"] = [start_y, end_y]

    return Profile(
        distances=distances,
        elevations=elevations,
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y,
        name=name,
        crs=crs,
        metadata=meta,
    )


def profile_from_coordinates(
    data: xr.DataArray,
    coordinates: Sequence[tuple[float, float]],
    point_spacing: float | None = None,
    name: str | None = None,
    metadata: dict | None = None,
    method: str = "nearest",
) -> Profile:
    """
    Create a Profile from a list of (x, y) coordinates.

    By default, samples only at the given vertices. Use ``point_spacing``
    to interpolate along each segment at regular intervals.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    coordinates : list[tuple[float, float]]
        List of (x, y) coordinate pairs defining the path
    point_spacing : float, optional
        Spacing between sample points in metres. When provided, each segment
        is interpolated at this interval.
    name : str, optional
        Name for this profile
    metadata : dict, optional
        Additional metadata
    method : str, optional
        Interpolation method for sampling the grid: ``"nearest"`` (default),
        ``"linear"``, ``"cubic"``, ``"quadratic"``, ``"slinear"``, or
        ``"zero"``.  All methods except ``"nearest"`` use
        :meth:`xarray.DataArray.interp` and may produce NaN at grid edges
        or where the input contains NaN.

    Returns
    -------
    Profile

    Examples
    --------
    >>> coords = [(-10.0, 50.0), (-9.5, 50.5), (-9.0, 51.0)]
    >>> prof = profile_from_coordinates(data, coords, name="Custom Path")
    >>> prof = profile_from_coordinates(data, coords, point_spacing=1000.0)
    """
    _validate_method(method)
    if len(coordinates) < 2:
        raise ValueError(f"Need at least 2 coordinates, got {len(coordinates)}")
    if point_spacing is not None and point_spacing <= 0:
        raise ValueError(f"point_spacing must be positive, got {point_spacing}")

    x_dim, y_dim = get_dim_names(data)
    crs = get_crs(data)
    projected = crs is not None and crs.is_projected

    x_min, x_max = float(data[x_dim].min()), float(data[x_dim].max())
    y_min, y_max = float(data[y_dim].min()), float(data[y_dim].max())
    for i, (cx, cy) in enumerate(coordinates):
        if not (x_min <= cx <= x_max):
            raise ValueError(
                f"coordinates[{i}] {x_dim} ({cx}) is outside "
                f"DEM bounds [{x_min:.2f}, {x_max:.2f}]"
            )
        if not (y_min <= cy <= y_max):
            raise ValueError(
                f"coordinates[{i}] {y_dim} ({cy}) is outside "
                f"DEM bounds [{y_min:.2f}, {y_max:.2f}]"
            )

    start_x, start_y = coordinates[0]
    end_x, end_y = coordinates[-1]

    if point_spacing is not None:
        all_elevations = []
        all_distances = []
        cumulative_m = 0.0

        for i in range(len(coordinates) - 1):
            sx, sy = coordinates[i]
            ex, ey = coordinates[i + 1]

            n = _calculate_num_points(
                sx,
                sy,
                ex,
                ey,
                num_points=None,
                point_spacing=point_spacing,
                crs=crs,
            )
            elevations, seg_distances = _extract_profile_arrays(
                data,
                sx,
                sy,
                ex,
                ey,
                n,
                crs,
                method=method,
            )

            if i > 0:
                elevations = elevations[1:]
                seg_distances = seg_distances[1:]

            all_elevations.append(elevations)
            all_distances.append(seg_distances + cumulative_m)
            cumulative_m = all_distances[-1][-1]

        dist_array = np.concatenate(all_distances)
        elev_array = np.concatenate(all_elevations)
    else:
        dist_list = []
        elev_list = []
        cumulative_m = 0.0

        for i, (cx, cy) in enumerate(coordinates):
            elev = float(_sample_grid(data, {x_dim: cx, y_dim: cy}, method))
            elev_list.append(elev)

            if i == 0:
                dist_list.append(0.0)
            else:
                px, py = coordinates[i - 1]
                if projected:
                    cumulative_m += math.hypot(cx - px, cy - py)
                else:
                    geod = Geodesic.WGS84
                    result = geod.Inverse(py, px, cy, cx)
                    cumulative_m += result["s12"]
                dist_list.append(cumulative_m)

        dist_array = np.array(dist_list)
        elev_array = np.array(elev_list)

    meta = dict(metadata) if metadata else {}
    meta["path_xs"] = [c[0] for c in coordinates]
    meta["path_ys"] = [c[1] for c in coordinates]

    return Profile(
        distances=dist_array,
        elevations=elev_array,
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y,
        name=name,
        crs=crs,
        metadata=meta,
    )


def cross_sections(
    data: xr.DataArray,
    profile: Profile,
    interval_m: float,
    section_width_m: float,
    num_points: int | None = None,
    point_spacing: float | None = None,
    method: str = "nearest",
) -> list[Profile]:
    """
    Create perpendicular cross-sections along a profile at regular intervals.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    profile : Profile
        The profile along which to create cross-sections
    interval_m : float
        Spacing between cross-sections in metres (must be positive)
    section_width_m : float
        Total width of each cross-section in metres (must be positive)
    num_points : int, optional
        Number of points along each cross-section
    point_spacing : float, optional
        Spacing between points in metres along cross-sections
    method : str, optional
        Interpolation method (default ``"nearest"``).  See
        :func:`extract_profile` for details.

    Returns
    -------
    list[Profile]

    Examples
    --------
    >>> prof = extract_profile(data, start=(-9.5, 52.0), end=(-5.5, 54.0))
    >>> sections = cross_sections(data, prof, interval_m=10000, section_width_m=20000)
    """
    if interval_m <= 0:
        raise ValueError(f"interval_m must be positive, got {interval_m}")
    if section_width_m <= 0:
        raise ValueError(f"section_width_m must be positive, got {section_width_m}")

    crs = profile.crs
    projected = crs is not None and crs.is_projected

    total_distance_m = profile.distances[-1]
    section_distances_m = np.arange(0, total_distance_m + interval_m, interval_m)
    if section_distances_m[-1] > total_distance_m:
        section_distances_m = section_distances_m[:-1]

    path_xs = profile.metadata.get("path_xs", [profile.start_x, profile.end_x])
    path_ys = profile.metadata.get("path_ys", [profile.start_y, profile.end_y])

    half_width_m = section_width_m / 2

    if projected:
        # Build Euclidean segments
        seg_cum_m = [0.0]
        for j in range(len(path_xs) - 1):
            seg_cum_m.append(
                seg_cum_m[-1]
                + math.hypot(path_xs[j + 1] - path_xs[j], path_ys[j + 1] - path_ys[j])
            )

        sections = []
        for i, dist_m in enumerate(section_distances_m):
            seg_idx = min(
                int(np.searchsorted(seg_cum_m, dist_m, side="right")) - 1,
                len(path_xs) - 2,
            )
            t = (
                (dist_m - seg_cum_m[seg_idx])
                / (seg_cum_m[seg_idx + 1] - seg_cum_m[seg_idx])
                if seg_cum_m[seg_idx + 1] != seg_cum_m[seg_idx]
                else 0.0
            )
            cx = path_xs[seg_idx] + t * (path_xs[seg_idx + 1] - path_xs[seg_idx])
            cy = path_ys[seg_idx] + t * (path_ys[seg_idx + 1] - path_ys[seg_idx])
            dx = path_xs[seg_idx + 1] - path_xs[seg_idx]
            dy = path_ys[seg_idx + 1] - path_ys[seg_idx]
            bearing = math.atan2(dx, dy)
            perp = bearing + math.pi / 2

            sx = cx + half_width_m * math.sin(perp)
            sy = cy + half_width_m * math.cos(perp)
            ex = cx - half_width_m * math.sin(perp)
            ey = cy - half_width_m * math.cos(perp)

            sections.append(
                extract_profile(
                    data,
                    start=(sx, sy),
                    end=(ex, ey),
                    num_points=num_points,
                    point_spacing=point_spacing,
                    name=f"Section_{i + 1}_at_{dist_m:.0f}m",
                    method=method,
                )
            )
    else:
        geod = Geodesic.WGS84

        seg_lines = []
        seg_cum_m = [0.0]
        for j in range(len(path_xs) - 1):
            seg = geod.InverseLine(
                path_ys[j], path_xs[j], path_ys[j + 1], path_xs[j + 1]
            )
            seg_lines.append(seg)
            seg_cum_m.append(seg_cum_m[-1] + seg.s13)

        sections = []
        for i, dist_m in enumerate(section_distances_m):
            seg_idx = min(
                int(np.searchsorted(seg_cum_m, dist_m, side="right")) - 1,
                len(seg_lines) - 1,
            )
            local_m = dist_m - seg_cum_m[seg_idx]
            pos = seg_lines[seg_idx].Position(local_m)
            center_x = pos["lon2"]
            center_y = pos["lat2"]
            local_bearing = pos["azi2"]

            perp_bearing = (local_bearing + 90) % 360

            start_result = geod.Direct(center_y, center_x, perp_bearing, half_width_m)
            end_result = geod.Direct(
                center_y, center_x, (perp_bearing + 180) % 360, half_width_m
            )

            sections.append(
                extract_profile(
                    data,
                    start=(start_result["lon2"], start_result["lat2"]),
                    end=(end_result["lon2"], end_result["lat2"]),
                    num_points=num_points,
                    point_spacing=point_spacing,
                    name=f"Section_{i + 1}_at_{dist_m:.0f}m",
                    method=method,
                )
            )

    return sections


def profiles_from_file(
    data: xr.DataArray,
    path: str | Path,
    id_column: str | None = None,
    point_spacing: float | None = None,
    method: str = "nearest",
) -> list[Profile]:
    """
    Create profiles from linestring features in a vector file.

    Accepts any format supported by GeoPandas (Shapefile, GeoPackage,
    GeoJSON, etc.).

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    path : str or Path
        Path to vector file containing LineString or MultiLineString features
    id_column : str, optional
        Column name to use for profile naming
    point_spacing : float, optional
        Spacing between sample points in metres. When provided, each segment
        is interpolated along the geodesic at this interval.
    method : str, optional
        Interpolation method (default ``"nearest"``).  See
        :func:`extract_profile` for details.

    Returns
    -------
    list[Profile]

    Examples
    --------
    >>> profiles = profiles_from_file(data, "canyons.gpkg", id_column="NAME")
    """
    return profiles_from_gdf(
        data,
        gpd.read_file(path),
        id_column=id_column,
        point_spacing=point_spacing,
        method=method,
    )


def profiles_from_gdf(
    data: xr.DataArray,
    gdf: gpd.GeoDataFrame,
    id_column: str | None = None,
    point_spacing: float | None = None,
    method: str = "nearest",
) -> list[Profile]:
    """
    Create profiles from LineString features in a GeoDataFrame.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with LineString or MultiLineString geometries
    id_column : str, optional
        Column to use for profile names
    point_spacing : float, optional
        Spacing between sample points in metres. When provided, each segment
        is interpolated along the geodesic at this interval.
    method : str, optional
        Interpolation method (default ``"nearest"``).  See
        :func:`extract_profile` for details.

    Returns
    -------
    list[Profile]

    Examples
    --------
    >>> profiles = profiles_from_gdf(data, gdf, id_column="name")
    """
    data_crs = get_crs(data)
    projected = data_crs is not None and data_crs.is_projected

    if projected:
        if gdf.crs is not None and gdf.crs != data_crs:
            raise ValueError(
                f"GeoDataFrame CRS ({gdf.crs}) does not match data CRS ({data_crs})"
            )
    else:
        if gdf.crs is not None and not gdf.crs.is_geographic:
            gdf = gdf.to_crs("EPSG:4326")

    x_dim, y_dim = get_dim_names(data)
    profiles = []
    skipped = 0
    x_min, x_max = float(data[x_dim].min()), float(data[x_dim].max())
    y_min, y_max = float(data[y_dim].min()), float(data[y_dim].max())

    for seq, (_, row) in enumerate(gdf.iterrows(), start=1):
        geom = row.geometry
        attributes = row.drop("geometry").to_dict()

        linestrings = []
        if geom.geom_type == "LineString":
            linestrings.append(geom)
        elif geom.geom_type == "MultiLineString":
            linestrings.extend(geom.geoms)
        else:
            logger.warning(
                f"Skipping feature {seq}: unsupported geometry type {geom.geom_type}"
            )
            skipped += 1
            continue

        for sub_idx, line in enumerate(linestrings):
            coords = [(c[0], c[1]) for c in line.coords]

            within_bounds = all(
                x_min <= cx <= x_max and y_min <= cy <= y_max for cx, cy in coords
            )
            if not within_bounds:
                skipped += 1
                continue

            if id_column and id_column in attributes:
                name = str(attributes[id_column])
            else:
                name = f"Feature_{seq}"
                if len(linestrings) > 1:
                    name += f"_Part_{sub_idx + 1}"

            meta = dict(attributes)
            if len(linestrings) > 1:
                meta["sub_index"] = sub_idx

            profiles.append(
                profile_from_coordinates(
                    data=data,
                    coordinates=coords,
                    point_spacing=point_spacing,
                    name=name,
                    metadata=meta,
                    method=method,
                )
            )

    if skipped > 0:
        logger.warning(
            f"Skipped {skipped} feature(s) outside DEM bounds or with "
            f"unsupported geometry"
        )

    return profiles


# ============================================================================
# Analysis functions
# ============================================================================


def profile_stats(profile: Profile) -> pl.DataFrame:
    """
    Calculate statistics for the profile.

    Parameters
    ----------
    profile : Profile

    Returns
    -------
    pl.DataFrame
        DataFrame with statistics
    """
    return pl.DataFrame(
        {
            "statistic": [
                "total_distance_m",
                "min_elevation_m",
                "max_elevation_m",
                "mean_elevation_m",
                "median_elevation_m",
                "std_elevation_m",
                "elevation_range_m",
            ],
            "value": [
                float(profile.distances[-1]),
                float(np.min(profile.elevations)),
                float(np.max(profile.elevations)),
                float(np.mean(profile.elevations)),
                float(np.median(profile.elevations)),
                float(np.std(profile.elevations)),
                float(np.max(profile.elevations) - np.min(profile.elevations)),
            ],
        }
    )


def max_depth(profile: Profile) -> tuple[float, float]:
    """
    Find the maximum depth point.

    Parameters
    ----------
    profile : Profile

    Returns
    -------
    tuple[float, float]
        (distance_m, depth_m) of the deepest point
    """
    idx = np.argmin(profile.elevations)
    return float(profile.distances[idx]), float(profile.elevations[idx])


def gradient(profile: Profile) -> np.ndarray:
    """
    Calculate the slope along the profile in degrees.

    Parameters
    ----------
    profile : Profile

    Returns
    -------
    np.ndarray
        Slope in degrees
    """
    grad = np.gradient(profile.elevations, profile.distances)
    return np.degrees(np.arctan(grad))


def concavity_index(profile: Profile) -> float:
    """
    Calculate Normalized Concavity Index (NCI) of the profile.

    Parameters
    ----------
    profile : Profile

    Returns
    -------
    float
        Positive = concave, negative = convex, near zero = straight

    Examples
    --------
    >>> nci = concavity_index(prof)
    """
    _, elevations = _ensure_descending(profile.distances, profile.elevations)
    reference_line = np.linspace(elevations[0], elevations[-1], len(elevations))
    deviations = elevations - reference_line
    median_deviation = np.median(deviations)
    relief = abs(elevations[-1] - elevations[0])

    if relief == 0:
        return 0.0

    return median_deviation / relief


def knickpoints(
    profile: Profile,
    threshold: float | None = None,
    smooth: float | None = None,
) -> pl.DataFrame:
    """
    Identify knickpoints (abrupt slope changes) along the profile.

    Parameters
    ----------
    profile : Profile
    threshold : float, optional
        Minimum rate of slope change (degrees/m). Defaults to 2 std above mean.
    smooth : float, optional
        Gaussian smoothing sigma before detection.

    Returns
    -------
    pl.DataFrame
        Knickpoints with columns: distance_m, depth_m, slope_break_deg
    """
    if smooth is not None and smooth <= 0:
        raise ValueError(f"smooth must be positive, got {smooth}")

    elevations = (
        gaussian_filter1d(profile.elevations, smooth) if smooth else profile.elevations
    )
    grad = np.gradient(elevations, profile.distances)
    slope_deg = np.degrees(np.arctan(np.abs(grad)))
    slope_break = np.abs(np.gradient(slope_deg, profile.distances))

    if threshold is None:
        threshold = np.mean(slope_break) + 2 * np.std(slope_break)

    peaks, properties = find_peaks(slope_break, height=threshold)

    return pl.DataFrame(
        {
            "distance_m": profile.distances[peaks],
            "depth_m": profile.elevations[peaks],
            "slope_break_deg": properties["peak_heights"],
        }
    )


def get_canyons(
    profile: Profile,
    prominence: float | None = None,
    smooth: float | None = None,
) -> pl.DataFrame:
    """
    Identify canyon features in the profile.

    Canyons are identified as prominent troughs bounded by peaks on both
    sides.  The **shoulder elevation** is the lower of the two bounding
    peaks; width, depth and cross-sectional area are all measured relative
    to that level.  Troughs without a detected peak on each side are
    skipped.

    Parameters
    ----------
    profile : Profile
    prominence : float, optional
        Minimum prominence (m) for canyon detection. Defaults to 10% of range.
    smooth : float, optional
        Gaussian smoothing sigma before detection.

    Returns
    -------
    pl.DataFrame
        Canyon measurements (distances in m, area in m²).
    """
    if smooth is not None and smooth <= 0:
        raise ValueError(f"smooth must be positive, got {smooth}")
    if prominence is not None and prominence <= 0:
        raise ValueError(f"prominence must be positive, got {prominence}")

    elevations = (
        gaussian_filter1d(profile.elevations, sigma=smooth)
        if smooth
        else profile.elevations.copy()
    )
    distances_m = profile.distances

    if prominence is None:
        prominence = (
            elevations.max() - elevations.min()
        ) * _DEFAULT_PROMINENCE_FRACTION

    peak_idx, _ = find_peaks(elevations, prominence=prominence)
    trough_idx, _ = find_peaks(-elevations, prominence=prominence)

    canyons = []
    for ti in trough_idx:
        left_peaks = peak_idx[peak_idx < ti]
        right_peaks = peak_idx[peak_idx > ti]
        if len(left_peaks) == 0 or len(right_peaks) == 0:
            logger.info(
                "Skipping trough at %.0f m: missing %s bounding peak",
                distances_m[ti],
                "left" if len(left_peaks) == 0 else "right",
            )
            continue
        li = left_peaks[-1]
        ri = right_peaks[0]

        shoulder_elev = min(elevations[li], elevations[ri])

        # Search outward from the trough on each side to find the first
        # crossing of the shoulder elevation, but only up to the bounding
        # peak so the width stays within the canyon walls.
        left_slice = slice(ti - 1, li - 1 if li > 0 else None, -1)
        right_slice = slice(ti + 1, ri + 1)
        width_start = _interp_crossing(
            elevations[left_slice],
            distances_m[left_slice],
            shoulder_elev,
        )
        width_end = _interp_crossing(
            elevations[right_slice],
            distances_m[right_slice],
            shoulder_elev,
        )
        if width_start is None or width_end is None:
            logger.info(
                "Skipping trough at %.0f m: shoulder crossing not found",
                distances_m[ti],
            )
            continue

        # Build area integral including the interpolated crossing points so
        # the edge slivers are not lost on coarse profiles.
        inside = (distances_m > width_start) & (distances_m < width_end)
        d_area = np.concatenate([[width_start], distances_m[inside], [width_end]])
        e_area = np.concatenate([[shoulder_elev], elevations[inside], [shoulder_elev]])
        depths = np.maximum(shoulder_elev - e_area, 0)

        canyons.append(
            {
                "floor_distance": float(distances_m[ti]),
                "floor_elevation": float(elevations[ti]),
                "shoulder_elevation": float(shoulder_elev),
                "width_start": float(width_start),
                "width_end": float(width_end),
                "width": float(width_end - width_start),
                "depth": float(shoulder_elev - elevations[ti]),
                "cross_sectional_area": float(trapezoid(depths, d_area)),
            }
        )

    if not canyons:
        return pl.DataFrame(
            schema={
                "floor_distance": pl.Float64,
                "floor_elevation": pl.Float64,
                "shoulder_elevation": pl.Float64,
                "width_start": pl.Float64,
                "width_end": pl.Float64,
                "width": pl.Float64,
                "depth": pl.Float64,
                "cross_sectional_area": pl.Float64,
            }
        )

    return pl.DataFrame(canyons)


# ============================================================================
# Multi-profile functions
# ============================================================================


def compare_stats(profiles: list[Profile]) -> pl.DataFrame:
    """
    Compare statistics across multiple profiles.

    Parameters
    ----------
    profiles : list[Profile]

    Returns
    -------
    pl.DataFrame
        Statistics for all profiles in wide format (profiles as columns)

    Examples
    --------
    >>> from bathy.profile import compare_stats
    >>> prof1 = extract_profile(data, start=(-8, 52), end=(-2, 58), name="Profile 1")
    >>> prof2 = extract_profile(data, start=(-8, 53), end=(-2, 59), name="Profile 2")
    >>> df = compare_stats([prof1, prof2])
    """
    if not profiles:
        raise ValueError("Need at least one profile to compare")

    all_stats = [profile_stats(prof) for prof in profiles]
    result: dict[str, list] = {"statistic": all_stats[0]["statistic"].to_list()}
    for i, (prof, stats) in enumerate(zip(profiles, all_stats), start=1):
        name = prof.name or f"Profile_{i}"
        result[name] = stats["value"].to_list()

    return pl.DataFrame(result)


def to_gdf(profiles: Profile | list[Profile]) -> gpd.GeoDataFrame:
    """
    Export one or more profiles as a GeoDataFrame.

    Parameters
    ----------
    profiles : Profile or list[Profile]

    Returns
    -------
    geopandas.GeoDataFrame
        One row per profile with LineString geometry and key statistics.

    Examples
    --------
    >>> from bathy.profile import to_gdf
    >>> gdf = to_gdf(prof)
    >>> gdf = to_gdf([prof1, prof2])
    >>> gdf.to_file("profiles.gpkg", driver="GPKG")
    """
    if isinstance(profiles, Profile):
        profiles = [profiles]
    if not profiles:
        raise ValueError("Need at least one profile")

    rows = []
    geometries = []
    for p in profiles:
        if "path_xs" not in p.metadata or "path_ys" not in p.metadata:
            raise ValueError(
                f"Profile {p.name!r} is missing path geometry in metadata. "
                "Create profiles using extract_profile() or profile_from_coordinates()."
            )
        row = {k: v for k, v in p.metadata.items() if not isinstance(v, (list, dict))}
        row.update(
            {
                "name": p.name,
                "total_distance_m": float(p.distances[-1]),
                "min_elevation_m": float(np.nanmin(p.elevations)),
                "max_elevation_m": float(np.nanmax(p.elevations)),
                "mean_elevation_m": float(np.nanmean(p.elevations)),
            }
        )
        rows.append(row)
        geometries.append(LineString(zip(p.metadata["path_xs"], p.metadata["path_ys"])))

    out_crs = profiles[0].crs or "EPSG:4326"
    return gpd.GeoDataFrame(rows, geometry=geometries, crs=out_crs)
