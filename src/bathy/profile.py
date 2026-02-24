"""Bathymetric profile functions."""

import logging
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import polars as pl
import xarray as xr
from geographiclib.geodesic import Geodesic
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from shapely.geometry import LineString

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
        Distances along profile (km)
    elevations : np.ndarray
        Elevation values along profile (m)
    start_lon, start_lat : float
        Starting coordinates
    end_lon, end_lat : float
        Ending coordinates
    name : str, optional
        Profile name
    metadata : dict
        Additional metadata (includes path_lons, path_lats for plotting)

    Examples
    --------
    >>> prof = extract_profile(data, start=(-9.5, 52.0), end=(-5.5, 52.0))
    >>> prof = extract_profile(
    ...     data, start=(-9.5, 52.0), end=(-5.5, 52.0), point_spacing=1.0
    ... )
    """

    distances: np.ndarray
    elevations: np.ndarray
    start_lon: float
    start_lat: float
    end_lon: float
    end_lat: float
    name: str | None = None
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        name = f'"{self.name}", ' if self.name else ""
        max_depth = np.nanmin(self.elevations)
        return (
            f"Profile({name}{self.distances[-1]:.1f} km, max_depth={max_depth:.0f} m)"
        )


# ============================================================================
# Internal helpers
# ============================================================================


def _validate_coordinates(
    data: xr.DataArray, lon: float, lat: float, param_name: str
) -> None:
    """Validate that coordinates are within data bounds."""
    lon_min, lon_max = float(data.lon.min()), float(data.lon.max())
    lat_min, lat_max = float(data.lat.min()), float(data.lat.max())

    if not (lon_min <= lon <= lon_max):
        raise ValueError(
            f"{param_name} longitude ({lon}) is outside "
            f"DEM bounds [{lon_min:.2f}, {lon_max:.2f}]"
        )
    if not (lat_min <= lat <= lat_max):
        raise ValueError(
            f"{param_name} latitude ({lat}) is outside "
            f"DEM bounds [{lat_min:.2f}, {lat_max:.2f}]"
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
        (distances, elevations) - flipped and re-zeroed if start elevation
        < end elevation
    """
    if elevations[0] < elevations[-1]:
        new_dist = distances[-1] - distances[::-1]
        return new_dist, elevations[::-1]
    return distances, elevations


def _calculate_num_points(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    num_points: int | None,
    point_spacing: float | None,
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

    if point_spacing <= 0:
        raise ValueError(f"point_spacing must be positive, got {point_spacing}")

    geod = Geodesic.WGS84
    result = geod.Inverse(start_lat, start_lon, end_lat, end_lon)
    total_distance_km = result["s12"] / 1000
    return max(2, int(np.ceil(total_distance_km / point_spacing)) + 1)


def _extract_profile_arrays(
    data: xr.DataArray,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract elevation and distance arrays along a straight-line profile.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (elevations, distances) arrays
    """
    lons = np.linspace(start_lon, end_lon, n)
    lats = np.linspace(start_lat, end_lat, n)

    lon_da = xr.DataArray(lons, dims="points")
    lat_da = xr.DataArray(lats, dims="points")
    elevations = data.sel(lon=lon_da, lat=lat_da, method="nearest").values.astype(float)

    geod = Geodesic.WGS84
    distances = np.zeros(n)
    for i in range(1, n):
        result = geod.Inverse(lats[i - 1], lons[i - 1], lats[i], lons[i])
        distances[i] = distances[i - 1] + result["s12"] / 1000

    return elevations, distances


def _find_crossing_m(
    elevations: np.ndarray,
    mask: np.ndarray,
    target_elev: float,
    distances_m: np.ndarray,
    fallback: float,
) -> float:
    """Find where profile crosses target elevation within masked region (metres)."""
    elevs, dists = elevations[mask], distances_m[mask]
    if len(elevs) == 0:
        return fallback
    return dists[np.argmin(np.abs(elevs - target_elev))]


def _normalise_profile(
    distances: np.ndarray, elevations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise distances and elevations to [0, 1]."""
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
) -> Profile:
    """
    Create a bathymetric profile between two points.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    start : tuple[float, float]
        Starting coordinates (lon, lat)
    end : tuple[float, float]
        Ending coordinates (lon, lat)
    num_points : int, optional
        Number of points along profile. Cannot be used with point_spacing.
        Default: 100 if neither num_points nor point_spacing is specified.
    point_spacing : float, optional
        Spacing between points in km. Cannot be used with num_points.
    name : str, optional
        Name for this profile
    metadata : dict, optional
        Additional metadata

    Returns
    -------
    Profile

    Examples
    --------
    >>> prof = extract_profile(data, start=(-9.5, 52.0), end=(-5.5, 52.0))
    >>> prof = extract_profile(
    ...     data, start=(-9.5, 52.0), end=(-5.5, 52.0), point_spacing=1.0
    ... )
    """
    start_lon, start_lat = start
    end_lon, end_lat = end

    _validate_coordinates(data, start_lon, start_lat, "start")
    _validate_coordinates(data, end_lon, end_lat, "end")

    n = _calculate_num_points(
        start_lon, start_lat, end_lon, end_lat, num_points, point_spacing
    )
    elevations, distances = _extract_profile_arrays(
        data, start_lon, start_lat, end_lon, end_lat, n
    )

    meta = dict(metadata) if metadata else {}
    meta["path_lons"] = [start_lon, end_lon]
    meta["path_lats"] = [start_lat, end_lat]

    return Profile(
        distances=distances,
        elevations=elevations,
        start_lon=start_lon,
        start_lat=start_lat,
        end_lon=end_lon,
        end_lat=end_lat,
        name=name,
        metadata=meta,
    )


def profile_from_coordinates(
    data: xr.DataArray,
    coordinates: list[tuple[float, float]],
    name: str | None = None,
    metadata: dict | None = None,
) -> Profile:
    """
    Create a Profile from a list of (lon, lat) coordinates.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    coordinates : list[tuple[float, float]]
        List of (lon, lat) coordinate pairs defining the path
    name : str, optional
        Name for this profile
    metadata : dict, optional
        Additional metadata

    Returns
    -------
    Profile

    Examples
    --------
    >>> coords = [(-10.0, 50.0), (-9.5, 50.5), (-9.0, 51.0)]
    >>> prof = profile_from_coordinates(data, coords, name="Custom Path")
    """
    if len(coordinates) < 2:
        raise ValueError(f"Need at least 2 coordinates, got {len(coordinates)}")

    start_lon, start_lat = coordinates[0]
    end_lon, end_lat = coordinates[-1]

    geod = Geodesic.WGS84
    dist_list = []
    elev_list = []
    cumulative_distance = 0.0

    for i, (lon, lat) in enumerate(coordinates):
        elev = float(data.sel(lon=lon, lat=lat, method="nearest").values)
        elev_list.append(elev)

        if i == 0:
            dist_list.append(0.0)
        else:
            prev_lon, prev_lat = coordinates[i - 1]
            result = geod.Inverse(prev_lat, prev_lon, lat, lon)
            cumulative_distance += result["s12"] / 1000
            dist_list.append(cumulative_distance)

    meta = dict(metadata) if metadata else {}
    meta["path_lons"] = [c[0] for c in coordinates]
    meta["path_lats"] = [c[1] for c in coordinates]

    return Profile(
        distances=np.array(dist_list),
        elevations=np.array(elev_list),
        start_lon=start_lon,
        start_lat=start_lat,
        end_lon=end_lon,
        end_lat=end_lat,
        name=name,
        metadata=meta,
    )


def cross_sections(
    data: xr.DataArray,
    profile: Profile,
    interval_km: float,
    section_width_km: float,
    num_points: int | None = None,
    point_spacing: float | None = None,
) -> list[Profile]:
    """
    Create perpendicular cross-sections along a profile at regular intervals.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    profile : Profile
        The profile along which to create cross-sections
    interval_km : float
        Spacing between cross-sections in kilometers (must be positive)
    section_width_km : float
        Total width of each cross-section in kilometers (must be positive)
    num_points : int, optional
        Number of points along each cross-section
    point_spacing : float, optional
        Spacing between points in km along cross-sections

    Returns
    -------
    list[Profile]

    Examples
    --------
    >>> prof = extract_profile(data, start=(-9.5, 52.0), end=(-5.5, 54.0))
    >>> sections = cross_sections(data, prof, interval_km=10, section_width_km=20)
    """
    if interval_km <= 0:
        raise ValueError(f"interval_km must be positive, got {interval_km}")
    if section_width_km <= 0:
        raise ValueError(f"section_width_km must be positive, got {section_width_km}")

    total_distance = profile.distances[-1]
    section_distances = np.arange(0, total_distance + interval_km, interval_km)
    if section_distances[-1] > total_distance:
        section_distances = section_distances[:-1]

    geod = Geodesic.WGS84

    line = geod.InverseLine(
        profile.start_lat, profile.start_lon, profile.end_lat, profile.end_lon
    )

    sections = []
    for i, dist_km in enumerate(section_distances):
        dist_m = dist_km * 1000

        pos = line.Position(dist_m)
        center_lon = pos["lon2"]
        center_lat = pos["lat2"]
        local_bearing = pos["azi2"]

        perp_bearing = (local_bearing + 90) % 360
        half_width_m = (section_width_km / 2) * 1000

        start_result = geod.Direct(center_lat, center_lon, perp_bearing, half_width_m)
        start_lon = start_result["lon2"]
        start_lat = start_result["lat2"]

        end_bearing = (perp_bearing + 180) % 360
        end_result = geod.Direct(center_lat, center_lon, end_bearing, half_width_m)
        end_lon = end_result["lon2"]
        end_lat = end_result["lat2"]

        section_name = f"Section_{i + 1}_at_{dist_km:.1f}km"
        sections.append(
            extract_profile(
                data,
                start=(start_lon, start_lat),
                end=(end_lon, end_lat),
                num_points=num_points,
                point_spacing=point_spacing,
                name=section_name,
            )
        )

    return sections


def profiles_from_shapefile(
    data: xr.DataArray,
    shapefile_path: str,
    id_column: str | None = None,
) -> list[Profile]:
    """
    Create profiles from linestring features in a shapefile.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    shapefile_path : str
        Path to shapefile containing LineString or MultiLineString features
    id_column : str, optional
        Column name to use for profile naming

    Returns
    -------
    list[Profile]

    Examples
    --------
    >>> profiles = profiles_from_shapefile(data, "canyons.shp", id_column="NAME")
    """
    return profiles_from_gdf(data, gpd.read_file(shapefile_path), id_column=id_column)


def profiles_from_gdf(
    data: xr.DataArray,
    gdf: gpd.GeoDataFrame,
    id_column: str | None = None,
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

    Returns
    -------
    list[Profile]

    Examples
    --------
    >>> profiles = profiles_from_gdf(data, gdf, id_column="name")
    """
    profiles = []
    skipped = 0
    lon_min, lon_max = float(data.lon.min()), float(data.lon.max())
    lat_min, lat_max = float(data.lat.min()), float(data.lat.max())

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

            within_bounds = any(
                lon_min <= lon <= lon_max and lat_min <= lat <= lat_max
                for lon, lat in coords
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
                    data=data, coordinates=coords, name=name, metadata=meta
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


def stats(profile: Profile) -> pl.DataFrame:
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
                "total_distance",
                "min_elevation",
                "max_elevation",
                "mean_elevation",
                "median_elevation",
                "std_elevation",
                "elevation_range",
            ],
            "value": [
                float(profile.distances[-1]),
                float(np.min(profile.elevations)),
                float(np.max(profile.elevations)),
                float(np.mean(profile.elevations)),
                float(np.median(profile.elevations)),
                float(np.std(profile.elevations)),
                float(np.ptp(profile.elevations)),
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
        (distance, depth) of the deepest point
    """
    idx = np.argmin(profile.elevations)
    return profile.distances[idx], profile.elevations[idx]


def gradient(profile: Profile) -> np.ndarray:
    """
    Calculate the gradient along the profile.

    Parameters
    ----------
    profile : Profile

    Returns
    -------
    np.ndarray
        Gradient values (m/km)
    """
    return np.gradient(profile.elevations, profile.distances)


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
    reference_line = np.linspace(
        profile.elevations[0], profile.elevations[-1], len(profile.elevations)
    )
    deviations = profile.elevations - reference_line
    median_deviation = np.median(deviations)
    relief = abs(profile.elevations[-1] - profile.elevations[0])

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
        Minimum rate of slope change (degrees/km). Defaults to 2 std above mean.
    smooth : float, optional
        Gaussian smoothing sigma before detection.

    Returns
    -------
    pl.DataFrame
        Knickpoints with columns: distance_km, depth_m, slope_break
    """
    elevations = (
        gaussian_filter1d(profile.elevations, smooth) if smooth else profile.elevations
    )
    grad = np.gradient(elevations, profile.distances * 1000)
    slope_deg = np.degrees(np.arctan(np.abs(grad)))
    slope_break = np.abs(np.gradient(slope_deg, profile.distances))

    if threshold is None:
        threshold = np.mean(slope_break) + 2 * np.std(slope_break)

    peaks, properties = find_peaks(slope_break, height=threshold)

    return pl.DataFrame(
        {
            "distance_km": profile.distances[peaks],
            "depth_m": profile.elevations[peaks],
            "slope_break": properties["peak_heights"],
        }
    )


def get_canyons(
    profile: Profile,
    prominence: float | None = None,
    smooth: float | None = None,
) -> pl.DataFrame:
    """
    Identify canyon features in the profile.

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
    distances_m = profile.distances * 1000

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
        li = left_peaks[-1] if len(left_peaks) else None
        ri = right_peaks[0] if len(right_peaks) else None

        if li is None and ri is None:
            continue

        if li is not None and ri is not None:
            lower_elev = min(elevations[li], elevations[ri])
        else:
            lower_elev = elevations[li] if li is not None else elevations[ri]

        if li is not None:
            width_start = distances_m[li]
        else:
            mask = np.arange(len(elevations)) < ti
            width_start = _find_crossing_m(
                elevations, mask, lower_elev, distances_m, distances_m[0]
            )

        if ri is not None:
            width_end = distances_m[ri]
        else:
            mask = np.arange(len(elevations)) > ti
            width_end = _find_crossing_m(
                elevations, mask, lower_elev, distances_m, distances_m[-1]
            )

        if li is not None and ri is not None:
            if elevations[li] < elevations[ri]:
                mask = (np.arange(len(elevations)) > ti) & (
                    np.arange(len(elevations)) <= ri
                )
                width_end = _find_crossing_m(
                    elevations, mask, lower_elev, distances_m, distances_m[ri]
                )
            elif elevations[ri] < elevations[li]:
                mask = (np.arange(len(elevations)) >= li) & (
                    np.arange(len(elevations)) < ti
                )
                width_start = _find_crossing_m(
                    elevations, mask, lower_elev, distances_m, distances_m[li]
                )

        area_mask = (distances_m >= width_start) & (distances_m <= width_end)
        depths = lower_elev - elevations[area_mask]
        depths = np.maximum(depths, 0)

        canyons.append(
            {
                "floor_distance": distances_m[ti],
                "floor_elevation": elevations[ti],
                "width_start": width_start,
                "width_end": width_end,
                "width": width_end - width_start,
                "depth": lower_elev - elevations[ti],
                "cross_sectional_area": trapezoid(depths, distances_m[area_mask]),
            }
        )

    if not canyons:
        return pl.DataFrame(
            schema={
                "floor_distance": pl.Float64,
                "floor_elevation": pl.Float64,
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

    all_stats = [stats(prof) for prof in profiles]
    result: dict[str, list] = {"statistic": all_stats[0]["statistic"].to_list()}
    for i, (prof, prof_stats) in enumerate(zip(profiles, all_stats), start=1):
        name = prof.name or f"Profile_{i}"
        result[name] = prof_stats["value"].to_list()

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
        CRS is EPSG:4326.

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
        if "path_lons" not in p.metadata or "path_lats" not in p.metadata:
            raise ValueError(
                f"Profile {p.name!r} is missing path geometry in metadata. "
                "Create profiles using extract_profile() or profile_from_coordinates()."
            )
        row = {k: v for k, v in p.metadata.items() if not isinstance(v, (list, dict))}
        row.update(
            {
                "name": p.name,
                "total_distance_km": float(p.distances[-1]),
                "min_elevation_m": float(np.nanmin(p.elevations)),
                "max_elevation_m": float(np.nanmax(p.elevations)),
                "mean_elevation_m": float(np.nanmean(p.elevations)),
            }
        )
        rows.append(row)
        geometries.append(
            LineString(zip(p.metadata["path_lons"], p.metadata["path_lats"]))
        )

    return gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")
