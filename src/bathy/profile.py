"""Profile class for bathymetric profile analysis."""

import logging

import cmocean.cm as cmo
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xarray as xr
from geographiclib.geodesic import Geodesic
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from bathy.utils import get_extent

logger = logging.getLogger(__name__)

# Default prominence threshold as fraction of total relief
_DEFAULT_PROMINENCE_FRACTION = 0.1


class Profile:
    """
    Single bathymetric profile for analysis.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    start_lon, start_lat : float
        Starting coordinates
    end_lon, end_lat : float
        Ending coordinates
    num_points : int, optional
        Number of points along profile. Cannot be used with point_spacing.
        Default: 100 if neither num_points nor point_spacing is specified.
    point_spacing : float, optional
        Spacing between points in km. Cannot be used with num_points.
        If specified, num_points will be calculated automatically.
    name : str, optional
        Name for this profile

    Attributes
    ----------
    distances : np.ndarray
        Distances along profile (km)
    elevations : np.ndarray
        Elevation values along profile (m)
    name : str
        Profile name
    num_points : int
        Actual number of points in the profile

    Examples
    --------
    Create a profile with 100 points (default):

    >>> from bathy import Bathymetry
    >>> bath = Bathymetry('gebco.nc', lon_range=(-10, -5))
    >>> prof = bath.profile(-9.5, 52.0, -5.5, 52.0, name="Profile A")

    Create a profile with a point every 1 km:

    >>> prof = bath.profile(-9.5, 52.0, -5.5, 52.0, point_spacing=1.0, name="Profile B")

    Analyze the profile:

    >>> prof.stats()
    >>> prof.max_depth()
    >>> prof.plot()
    """

    def __init__(
        self,
        data: xr.DataArray,
        start_lon: float,
        start_lat: float,
        end_lon: float,
        end_lat: float,
        num_points: int | None = None,
        point_spacing: float | None = None,
        name: str | None = None,
        metadata: dict | None = None,
    ):
        self.data = data
        self.start_lon = start_lon
        self.start_lat = start_lat
        self.end_lon = end_lon
        self.end_lat = end_lat
        self.name = name
        self.metadata = metadata or {}

        # Validate coordinates
        self._validate_coordinates(data, start_lon, start_lat, "start")
        self._validate_coordinates(data, end_lon, end_lat, "end")

        # Determine number of points
        self.num_points = self._calculate_num_points(start_lon, start_lat, end_lon, end_lat, num_points, point_spacing)

        # Extract profile elevations and distances
        self.elevations, self.distances = self._extract_profile()

    @classmethod
    def from_coordinates(
        cls,
        data: xr.DataArray,
        coordinates: list[tuple[float, float]],
        name: str | None = None,
        metadata: dict | None = None,
    ) -> "Profile":
        """
        Create a Profile from a list of (lon, lat) coordinates.

        This constructor allows creating profiles that follow arbitrary paths
        (e.g., from shapefiles) rather than straight lines.

        Parameters
        ----------
        data : xr.DataArray
            Elevation data
        coordinates : list[tuple[float, float]]
            List of (lon, lat) coordinate pairs defining the path
        name : str, optional
            Name for this profile
        metadata : dict, optional
            Additional metadata to store with the profile

        Returns
        -------
        Profile
            Profile object following the given coordinates

        Examples
        --------
        >>> coords = [(-10.0, 50.0), (-9.5, 50.5), (-9.0, 51.0)]
        >>> prof = Profile.from_coordinates(data, coords, name="Custom Path")
        """
        if len(coordinates) < 2:
            raise ValueError(f"Need at least 2 coordinates, got {len(coordinates)}")

        # Extract start and end
        start_lon, start_lat = coordinates[0]
        end_lon, end_lat = coordinates[-1]

        # Create a Profile instance bypassing normal __init__
        profile = cls.__new__(cls)
        profile.data = data
        profile.start_lon = start_lon
        profile.start_lat = start_lat
        profile.end_lon = end_lon
        profile.end_lat = end_lat
        profile.name = name
        profile.metadata = metadata or {}

        # Store the full path for plotting
        profile.metadata["path_lons"] = [c[0] for c in coordinates]
        profile.metadata["path_lats"] = [c[1] for c in coordinates]

        # Sample elevations and calculate cumulative distances
        geod = Geodesic.WGS84
        distances = []
        elevations = []
        cumulative_distance = 0.0

        for i, (lon, lat) in enumerate(coordinates):
            # Add elevation at this point
            elev = float(data.sel(lon=lon, lat=lat, method="nearest").values)
            elevations.append(elev)

            # Add cumulative distance
            if i == 0:
                distances.append(0.0)
            else:
                prev_lon, prev_lat = coordinates[i - 1]
                result = geod.Inverse(prev_lat, prev_lon, lat, lon)
                cumulative_distance += result["s12"] / 1000
                distances.append(cumulative_distance)

        profile.num_points = len(coordinates)
        profile.distances = np.array(distances)
        profile.elevations = np.array(elevations)

        return profile

    @classmethod
    def cross_sections(
        cls,
        data: xr.DataArray,
        profile: "Profile",
        interval_km: float,
        section_width_km: float,
        num_points: int | None = None,
        point_spacing: float | None = None,
    ) -> list["Profile"]:
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
            Total width of each cross-section in kilometers (half on each side, must be positive)
        num_points : int, optional
            Number of points along each cross-section
        point_spacing : float, optional
            Spacing between points in km along cross-sections

        Returns
        -------
        list[Profile]
            List of Profile objects representing perpendicular cross-sections

        Examples
        --------
        >>> prof = bath.profile(-9.5, 52.0, -5.5, 54.0)
        >>> sections = Profile.cross_sections(bath.data, prof, interval_km=10, section_width_km=20)
        """
        # Validate parameters
        if interval_km <= 0:
            raise ValueError(f"interval_km must be positive, got {interval_km}")
        if section_width_km <= 0:
            raise ValueError(f"section_width_km must be positive, got {section_width_km}")

        # Calculate total distance
        total_distance = profile.distances[-1]

        # Generate section positions
        section_distances = np.arange(0, total_distance + interval_km, interval_km)
        if section_distances[-1] > total_distance:
            section_distances = section_distances[:-1]

        geod = Geodesic.WGS84

        # Calculate the bearing of the main profile
        result = geod.Inverse(profile.start_lat, profile.start_lon, profile.end_lat, profile.end_lon)
        profile_bearing = result["azi1"]

        # Create cross-sections
        sections = []
        for i, dist_km in enumerate(section_distances):
            # Find position along profile at this distance
            dist_m = dist_km * 1000

            # Get position along the great circle
            line_result = geod.InverseLine(profile.start_lat, profile.start_lon, profile.end_lat, profile.end_lon)
            pos = line_result.Position(dist_m)
            center_lon = pos["lon2"]
            center_lat = pos["lat2"]

            # Calculate perpendicular bearing (90 degrees from profile bearing)
            perp_bearing = (profile_bearing + 90) % 360

            # Calculate endpoints of cross-section
            half_width_m = (section_width_km / 2) * 1000

            # Start point (perpendicular bearing)
            start_result = geod.Direct(center_lat, center_lon, perp_bearing, half_width_m)
            start_lon = start_result["lon2"]
            start_lat = start_result["lat2"]

            # End point (opposite direction)
            end_bearing = (perp_bearing + 180) % 360
            end_result = geod.Direct(center_lat, center_lon, end_bearing, half_width_m)
            end_lon = end_result["lon2"]
            end_lat = end_result["lat2"]

            # Create profile for this cross-section
            section_name = f"Section_{i + 1}_at_{dist_km:.1f}km"
            section = cls(
                data, start_lon, start_lat, end_lon, end_lat, num_points=num_points, point_spacing=point_spacing, name=section_name
            )
            sections.append(section)

        return sections

    @classmethod
    def from_shapefile(
        cls,
        data: xr.DataArray,
        shapefile_path: str,
        id_column: str | None = None,
    ) -> list["Profile"]:
        """
        Create profiles from linestring features in a shapefile.

        Each LineString feature becomes one profile following the feature's path.
        MultiLineString features are split into separate profiles.

        Parameters
        ----------
        data : xr.DataArray
            Elevation data
        shapefile_path : str
            Path to shapefile containing LineString or MultiLineString features
        id_column : str, optional
            Column name to use for profile naming/ID. If None, uses sequential numbering.

        Returns
        -------
        list[Profile]
            List of Profile objects, one for each linestring

        Examples
        --------
        >>> profiles = Profile.from_shapefile(bath.data, "canyons.shp", id_column="NAME")
        """
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError("geopandas is required for shapefile support. Install with: pip install geopandas")

        profiles = []
        skipped = 0

        # Read shapefile using geopandas
        gdf = gpd.read_file(shapefile_path)

        for idx, row in gdf.iterrows():
            geom = row.geometry
            attributes = row.drop("geometry").to_dict()

            # Extract linestrings
            linestrings = []
            if geom.geom_type == "LineString":
                linestrings.append(geom)
            elif geom.geom_type == "MultiLineString":
                linestrings.extend(geom.geoms)
            else:
                logger.warning(f"Skipping feature {idx}: unsupported geometry type {geom.geom_type}")
                skipped += 1
                continue

            # Create profile for each linestring
            for sub_idx, line in enumerate(linestrings):
                coords = list(line.coords)

                # Check if coordinates are within DEM bounds
                lon_min, lon_max = float(data.lon.min()), float(data.lon.max())
                lat_min, lat_max = float(data.lat.min()), float(data.lat.max())

                within_bounds = any(lon_min <= lon <= lon_max and lat_min <= lat <= lat_max for lon, lat in coords)

                if not within_bounds:
                    skipped += 1
                    continue

                # Create profile name
                if id_column and id_column in attributes:
                    name = str(attributes[id_column])
                else:
                    name = f"Feature_{idx + 1}"
                    if len(linestrings) > 1:
                        name += f"_Part_{sub_idx + 1}"

                # Add sub_index to metadata for MultiLineStrings
                if len(linestrings) > 1:
                    attributes["sub_index"] = sub_idx

                # Use Profile.from_coordinates classmethod
                profile = cls.from_coordinates(data=data, coordinates=coords, name=name, metadata=attributes)
                profiles.append(profile)

        if skipped > 0:
            logger.warning(f"Skipped {skipped} feature(s) outside DEM bounds or with unsupported geometry")

        return profiles

    @staticmethod
    def _validate_coordinates(data: xr.DataArray, lon: float, lat: float, param_name: str) -> None:
        """Validate that coordinates are within data bounds."""
        lon_min, lon_max = float(data.lon.min()), float(data.lon.max())
        lat_min, lat_max = float(data.lat.min()), float(data.lat.max())

        if not (lon_min <= lon <= lon_max):
            raise ValueError(f"{param_name} longitude ({lon}) is outside DEM bounds [{lon_min:.2f}, {lon_max:.2f}]")
        if not (lat_min <= lat <= lat_max):
            raise ValueError(f"{param_name} latitude ({lat}) is outside DEM bounds [{lat_min:.2f}, {lat_max:.2f}]")

    @staticmethod
    def _ensure_descending(distances: np.ndarray, elevations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Ensure profile descends from higher to lower elevation values.

        For bathymetry (negative elevations): -100m > -4000m, so profile runs shallow to deep.
        For topography (positive elevations): 1000m > 100m, so profile runs high to low.

        Parameters
        ----------
        distances : np.ndarray
            Distance values along profile
        elevations : np.ndarray
            Elevation values along profile

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (distances, elevations) - flipped if start elevation < end elevation

        Examples
        --------
        Bathymetry: start=-4000m, end=-100m → flips to start=-100m (shallow to deep)
        Topography: start=100m, end=1000m → flips to start=1000m (high to low)
        """
        if elevations[0] < elevations[-1]:
            return distances[::-1], elevations[::-1]
        return distances, elevations

    @staticmethod
    def _calculate_num_points(
        start_lon: float,
        start_lat: float,
        end_lon: float,
        end_lat: float,
        num_points: int | None,
        point_spacing: float | None,
    ) -> int:
        """Calculate number of points for the profile."""
        # Validate that exactly one of num_points or point_spacing is provided
        if num_points is None and point_spacing is None:
            return 100  # Default value
        if num_points is not None and point_spacing is not None:
            raise ValueError("Cannot specify both num_points and point_spacing. Choose one.")

        # Use num_points if provided
        if num_points is not None:
            if num_points < 1:
                raise ValueError(f"num_points must be at least 1, got {num_points}")
            return num_points

        # Calculate num_points from point_spacing
        if point_spacing <= 0:
            raise ValueError(f"point_spacing must be positive, got {point_spacing}")

        geod = Geodesic.WGS84
        result = geod.Inverse(start_lat, start_lon, end_lat, end_lon)
        total_distance_km = result["s12"] / 1000
        return max(2, int(np.ceil(total_distance_km / point_spacing)) + 1)

    def _extract_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract elevation and distance data along the profile line.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (elevations, distances) arrays
        """
        # Generate equally-spaced coordinates along the profile
        lons = np.linspace(self.start_lon, self.end_lon, self.num_points)
        lats = np.linspace(self.start_lat, self.end_lat, self.num_points)

        # Extract elevations at each point
        elevations = np.array([float(self.data.sel(lon=lon, lat=lat, method="nearest").values) for lon, lat in zip(lons, lats)])

        # Calculate cumulative distances
        geod = Geodesic.WGS84
        distances = np.zeros(self.num_points)
        for i in range(1, self.num_points):
            result = geod.Inverse(lats[i - 1], lons[i - 1], lats[i], lons[i])
            distances[i] = distances[i - 1] + result["s12"] / 1000

        return elevations, distances

    def stats(self) -> pl.DataFrame:
        """
        Calculate statistics for the profile.

        Returns
        -------
        pl.DataFrame
            DataFrame with statistics including total_distance, min/max/mean/median/std elevation, and elevation_range
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
                    float(self.distances[-1]),
                    float(np.min(self.elevations)),
                    float(np.max(self.elevations)),
                    float(np.mean(self.elevations)),
                    float(np.median(self.elevations)),
                    float(np.std(self.elevations)),
                    float(np.ptp(self.elevations)),
                ],
            }
        )

    def max_depth(self) -> tuple[float, float]:
        """
        Find the maximum depth point (distance in km, depth in m).

        Returns
        -------
        tuple[float, float]
            (distance, depth) of the deepest point
        """
        idx = np.argmin(self.elevations)
        return self.distances[idx], self.elevations[idx]

    def gradient(self) -> np.ndarray:
        """
        Calculate the gradient along the profile.

        Returns
        -------
        np.ndarray
            Gradient values (m/km)
        """
        return np.gradient(self.elevations, self.distances)

    def concavity_index(self) -> float:
        """
        Calculate Normalized Concavity Index (NCI) of the profile.

        The NCI measures profile concavity by comparing the actual profile to a straight
        line between start and end points. It calculates the median vertical deviation
        normalized by the total relief (elevation change).

        Returns
        -------
        float
            Normalized Concavity Index:
            - Positive values: concave profile (actual below straight line)
            - Negative values: convex profile (actual above straight line)
            - Near zero: approximately straight profile

        Notes
        -----
        Calculation:
        1. Fit a straight line between start and end points
        2. Calculate vertical deviations (actual - straight line) at all points
        3. Find the median deviation
        4. Normalize by total relief (|end_elevation - start_elevation|)

        Examples
        --------
        >>> nci = prof.concavity_index()
        >>> print(f"NCI: {nci:.3f}")
        """
        # Straight line from start to end
        reference_line = np.linspace(self.elevations[0], self.elevations[-1], len(self.elevations))

        # Vertical deviations (actual - reference)
        deviations = self.elevations - reference_line

        # Median deviation
        median_deviation = np.median(deviations)

        # Total relief (elevation change from start to end)
        relief = abs(self.elevations[-1] - self.elevations[0])

        # Avoid division by zero
        if relief == 0:
            return 0.0

        # Normalize by relief
        return median_deviation / relief

    def plot_gradient(self, **kwargs):
        """
        Plot the gradient (derivative) along the profile.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to plot

        Returns
        -------
        Figure, Axes
            Matplotlib figure and axes
        """
        grad = self.gradient()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self.distances, grad, **kwargs)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Gradient (m km⁻¹)")
        ax.grid(True, alpha=0.3)

        return fig, ax

    def get_canyons(self, prominence: float | None = None, smooth: float | None = None) -> pl.DataFrame:
        """
        Identify canyon features in the profile.

        Parameters
        ----------
        prominence : float, optional
            Minimum prominence (m) for canyon detection. If None, uses 10% of elevation range.
        smooth : float, optional
            Gaussian smoothing sigma before detection. Typical values: 2-5.

        Returns
        -------
        pl.DataFrame
            Canyon measurements (all distances in m, area in m²).
        """
        if smooth is not None and smooth <= 0:
            raise ValueError(f"smooth must be positive, got {smooth}")
        if prominence is not None and prominence <= 0:
            raise ValueError(f"prominence must be positive, got {prominence}")

        elevations = gaussian_filter1d(self.elevations, sigma=smooth) if smooth else self.elevations
        distances_m = self.distances * 1000  # Work in metres throughout

        if prominence is None:
            prominence = (elevations.max() - elevations.min()) * _DEFAULT_PROMINENCE_FRACTION

        peak_idx, _ = find_peaks(elevations, prominence=prominence)
        trough_idx, _ = find_peaks(-elevations, prominence=prominence)

        canyons = []
        for ti in trough_idx:
            left_peaks = peak_idx[peak_idx < ti]
            right_peaks = peak_idx[peak_idx > ti]
            li = left_peaks[-1] if len(left_peaks) else None
            ri = right_peaks[0] if len(right_peaks) else None

            if li is None and ri is None:
                continue  # Skip troughs with no shoulders

            # Determine lower shoulder elevation
            if li is not None and ri is not None:
                lower_elev = min(self.elevations[li], self.elevations[ri])
            else:
                lower_elev = self.elevations[li] if li is not None else self.elevations[ri]

            # Find width endpoints at lower shoulder elevation
            if li is not None:
                width_start = distances_m[li]
            else:
                mask = np.arange(len(elevations)) < ti
                width_start = self._find_crossing_m(mask, lower_elev, distances_m, distances_m[0])

            if ri is not None:
                width_end = distances_m[ri]
            else:
                mask = np.arange(len(elevations)) > ti
                width_end = self._find_crossing_m(mask, lower_elev, distances_m, distances_m[-1])

            # If one shoulder is higher, find where its elevation crosses the opposite slope
            if li is not None and ri is not None:
                if self.elevations[li] < self.elevations[ri]:
                    mask = (np.arange(len(elevations)) > ti) & (np.arange(len(elevations)) <= ri)
                    width_end = self._find_crossing_m(mask, lower_elev, distances_m, distances_m[ri])
                elif self.elevations[ri] < self.elevations[li]:
                    mask = (np.arange(len(elevations)) >= li) & (np.arange(len(elevations)) < ti)
                    width_start = self._find_crossing_m(mask, lower_elev, distances_m, distances_m[li])

            # Cross-sectional area
            area_mask = (distances_m >= width_start) & (distances_m <= width_end)
            depths = lower_elev - self.elevations[area_mask]
            depths = np.maximum(depths, 0)

            canyons.append(
                {
                    "floor_distance": distances_m[ti],
                    "floor_elevation": self.elevations[ti],
                    "width_start": width_start,
                    "width_end": width_end,
                    "width": width_end - width_start,
                    "depth": lower_elev - self.elevations[ti],
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

    def _find_crossing_m(self, mask: np.ndarray, target_elev: float, distances_m: np.ndarray, fallback: float) -> float:
        """Find where profile crosses target elevation within masked region (returns metres)."""
        elevs, dists = self.elevations[mask], distances_m[mask]
        if len(elevs) == 0:
            return fallback
        return dists[np.argmin(np.abs(elevs - target_elev))]

    def plot_canyons(self, canyons: pl.DataFrame | None = None, prominence: float | None = None, smooth: float | None = None, **kwargs):
        """
        Plot profile with canyons marked.

        Parameters
        ----------
        canyons : pl.DataFrame, optional
            Canyon data from get_canyons(). If None, detects canyons using prominence/smooth.
        prominence : float, optional
            Minimum prominence (m) for canyon detection (ignored if canyons provided).
        smooth : float, optional
            Gaussian smoothing sigma before detection (ignored if canyons provided).
        **kwargs
            Additional arguments passed to profile.plot()

        Returns
        -------
        Figure, list[Axes]
            Matplotlib figure and axes.
        """
        if canyons is None:
            canyons = self.get_canyons(prominence=prominence, smooth=smooth)

        if len(canyons) == 0:
            logger.info("No canyons detected. Try adjusting prominence or smoothing.")
            return self.plot(smooth=smooth, **kwargs)

        fig, axes = self.plot(smooth=smooth, **kwargs)
        ax = axes[-1]

        for row in canyons.iter_rows(named=True):
            floor_km = row["floor_distance"] / 1000
            floor_elev = row["floor_elevation"]
            shoulder_elev = floor_elev + row["depth"]
            ws_km, we_km = row["width_start"] / 1000, row["width_end"] / 1000

            # Floor marker
            ax.plot(floor_km, floor_elev, "ro", markersize=8, zorder=10)

            # Width line at shoulder elevation
            ax.plot([ws_km, we_km], [shoulder_elev] * 2, "k--", linewidth=1.5, alpha=0.7, zorder=5)

            # Depth line
            ax.plot([floor_km] * 2, [floor_elev, shoulder_elev], "k--", linewidth=1.5, alpha=0.7, zorder=5)

        return fig, axes

    def plot(
        self,
        show_map: bool = False,
        smooth: float | None = None,
        normalize: bool = False,
        ensure_descending: bool = False,
        cmap=cmo.deep_r,
        **kwargs,
    ):
        """
        Plot the bathymetric profile.

        Parameters
        ----------
        show_map : bool
            If True, show map with profile line
        smooth : float, optional
            Gaussian smoothing sigma. Typical values: 2-5.
        normalize : bool
            If True, normalize elevation to 0-1 range
        ensure_descending : bool
            If True, orient profile to descend from higher to lower elevation (ignoring user-defined start/end).
            If False (default), distance axis starts at the user-defined start point.
        **kwargs
            Additional arguments passed to matplotlib plot()

        Returns
        -------
        Figure, list[Axes]
            Matplotlib figure and list of axes (2 axes if show_map=True, 1 axis otherwise)

        Examples
        --------
        >>> fig, axes = prof.plot()
        >>> fig, axes = prof.plot(smooth=2.0)
        >>> fig, axes = prof.plot(show_map=True)
        >>> fig, axes = prof.plot(normalize=True)
        """
        # Apply smoothing
        elevations = gaussian_filter1d(self.elevations, sigma=smooth) if smooth else self.elevations
        distances = self.distances.copy()

        # Optionally ensure profile descends from higher to lower elevation
        if ensure_descending:
            distances, elevations = self._ensure_descending(distances, elevations)

        # Normalize if requested
        if normalize:
            # Normalize elevation to 0-1
            elev_min, elev_max = float(elevations.min()), float(elevations.max())
            elevations = (elevations - elev_min) / (elev_max - elev_min)

            # Normalize distance to 0-1
            dist_min, dist_max = float(distances.min()), float(distances.max())
            distances = (distances - dist_min) / (dist_max - dist_min)

        # Set axis limits to exact data range (no padding)
        ylim = (float(elevations.min()), float(elevations.max()))
        xlim = (float(distances.min()), float(distances.max()))

        if show_map:
            fig, (ax_map, ax_profile) = plt.subplots(1, 2, figsize=(16, 6))

            # Plot map
            extent = get_extent(self.data)
            ax_map.imshow(self.data.values, cmap=cmap, origin="lower", extent=extent, aspect="auto")
            ax_map.plot([self.start_lon, self.end_lon], [self.start_lat, self.end_lat], "r-", linewidth=2, label="Profile line")
            ax_map.plot(self.start_lon, self.start_lat, "go", markersize=10, label="Start")
            ax_map.plot(self.end_lon, self.end_lat, "ro", markersize=10, label="End")
            ax_map.set_xlabel("Longitude (°)")
            ax_map.set_ylabel("Latitude (°)")
            ax_map.legend()
        else:
            fig, ax_profile = plt.subplots(figsize=(12, 5))

        # Plot profile
        ax_profile.plot(distances, elevations, **kwargs)
        ax_profile.fill_between(distances, elevations, elevations.min(), alpha=0.3)

        # Configure axes
        ax_profile.set_xlabel("Normalized distance" if normalize else "Distance (km)")
        ax_profile.set_ylabel("Normalized elevation" if normalize else "Elevation (m)")
        ax_profile.set_xlim(xlim)
        ax_profile.set_ylim(ylim)
        ax_profile.grid(True, alpha=0.3)

        if show_map:
            return fig, [ax_map, ax_profile]
        return fig, [ax_profile]


# ============================================================================
# Module-level functions for working with multiple profiles
# ============================================================================


def compare_stats(profiles: list[Profile]) -> pl.DataFrame:
    """
    Compare statistics across multiple profiles.

    Parameters
    ----------
    profiles : list[Profile]
        List of Profile objects to compare

    Returns
    -------
    pl.DataFrame
        Statistics for all profiles in wide format (profiles as columns)

    Examples
    --------
    >>> from bathy.profile import compare_stats
    >>> prof1 = bath.profile(-8, 52, -2, 58, name="Profile 1")
    >>> prof2 = bath.profile(-8, 53, -2, 59, name="Profile 2")
    >>> stats = compare_stats([prof1, prof2])
    """
    if not profiles:
        raise ValueError("Need at least one profile to compare")

    # Get stats for each profile
    all_stats = {}
    for prof in profiles:
        name = prof.name if prof.name else f"Profile_{profiles.index(prof) + 1}"
        stats_df = prof.stats()
        all_stats[name] = dict(zip(stats_df["statistic"], stats_df["value"]))

    # Get statistic names from first profile
    first_profile_name = list(all_stats.keys())[0]
    statistic_names = list(all_stats[first_profile_name].keys())

    # Build DataFrame with statistic column first, then profile columns
    data = {"statistic": statistic_names}
    for profile_name, stats in all_stats.items():
        data[profile_name] = [stats[stat] for stat in statistic_names]

    return pl.DataFrame(data)


def plot_profiles(
    profiles: Profile | list[Profile],
    show_map: bool = False,
    normalize: bool = False,
    ensure_descending: bool = False,
    cmap=cmo.deep_r,
    **kwargs,
):
    """
    Plot multiple profiles on the same axes.

    Parameters
    ----------
    profiles : Profile or list[Profile]
        Profile object(s) to plot
    show_map : bool
        If True, show map with profile lines alongside the profile plot
    normalize : bool
        If True, normalize each profile's elevation and distance to 0-1
    ensure_descending : bool
        If True, orient profiles to descend from higher to lower elevation (ignoring user-defined start/end).
        If False (default), distance axis starts at the user-defined start point.
    **kwargs
        Additional arguments passed to matplotlib plot()

    Returns
    -------
    Figure, list[Axes]
        Matplotlib figure and list of axes (2 axes if show_map=True, 1 axis otherwise)

    Examples
    --------
    >>> from bathy.profile import plot_profiles
    >>> prof1 = bath.profile(start=(-8, 52), end=(-2, 58), name="Profile 1")
    >>> prof2 = bath.profile(start=(-8, 53), end=(-2, 59), name="Profile 2")
    >>> plot_profiles([prof1, prof2])
    >>> # With map
    >>> plot_profiles([prof1, prof2], show_map=True)
    >>> # Single profile
    >>> plot_profiles(prof1)
    """
    # Handle single profile
    if isinstance(profiles, Profile):
        profiles = [profiles]

    if not profiles:
        raise ValueError("Need at least one profile to plot")

    # Create figure with optional map
    if show_map:
        fig, (ax_map, ax_profile) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot map with profile lines
        bathymetry_data = profiles[0].data
        extent = get_extent(bathymetry_data)
        ax_map.imshow(bathymetry_data.values, cmap=cmap, origin="lower", extent=extent, aspect="auto", alpha=0.6)

        for i, prof in enumerate(profiles, start=1):
            label = prof.name if prof.name else f"Profile {i}"
            ax_map.plot([prof.start_lon, prof.end_lon], [prof.start_lat, prof.end_lat], "-", linewidth=2, label=label)
            ax_map.plot(prof.start_lon, prof.start_lat, "o", markersize=6)
            ax_map.plot(prof.end_lon, prof.end_lat, "s", markersize=6)

        ax_map.set_xlabel("Longitude (°)")
        ax_map.set_ylabel("Latitude (°)")
        ax_map.legend()
    else:
        fig, ax_profile = plt.subplots(figsize=(12, 6))

    for i, prof in enumerate(profiles, start=1):
        distances = prof.distances.copy()
        elevations = prof.elevations.copy()

        # Optionally ensure profile descends from higher to lower elevation
        if ensure_descending:
            distances, elevations = Profile._ensure_descending(distances, elevations)

        if normalize:
            # Normalize elevation to 0-1
            elev_min, elev_max = float(elevations.min()), float(elevations.max())
            if elev_max > elev_min:
                elevations = (elevations - elev_min) / (elev_max - elev_min)

            # Normalize distance to 0-1
            dist_min, dist_max = float(distances.min()), float(distances.max())
            if dist_max > dist_min:
                distances = (distances - dist_min) / (dist_max - dist_min)

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
):
    """
    Plot multiple profiles in a grid of subplots.

    Parameters
    ----------
    profiles : Profile or list[Profile]
        Profile object(s) to plot
    cols : int
        Number of columns in the grid (default: 2)
    figsize : tuple[float, float], optional
        Figure size (width, height). If None, calculated based on grid size.
    main_profile : Profile, optional
        Optional main profile (e.g., for cross-sections). If provided, a vertical line
        marks where the main profile intersects each cross-section.
    smooth : float, optional
        Gaussian smoothing sigma. Typical values: 2-5.
    normalize : bool
        If True, normalize each profile's elevation and distance to 0-1
    ensure_descending : bool
        If True, orient profiles to descend from higher to lower elevation (ignoring user-defined start/end).
        If False (default), distance axis starts at the user-defined start point.
    **kwargs
        Additional arguments passed to matplotlib plot()

    Returns
    -------
    Figure, np.ndarray
        Matplotlib figure and array of axes

    Examples
    --------
    >>> from bathy.profile import plot_profiles_grid
    >>> profiles = Profile.from_shapefile(bath.data, "canyons.shp")
    >>> plot_profiles_grid(profiles[:10])
    >>> # With smoothing
    >>> plot_profiles_grid(profiles[:10], smooth=3.0)
    >>> # With cross-sections and main profile
    >>> main = bath.profile(start=(-9.5, 52.0), end=(-5.5, 54.0))
    >>> sections = Profile.cross_sections(bath.data, main, interval_km=20, section_width_km=30)
    >>> plot_profiles_grid(sections, main_profile=main)
    """
    # Handle single profile
    if isinstance(profiles, Profile):
        profiles = [profiles]

    if not profiles:
        raise ValueError("Need at least one profile to plot")

    # Calculate grid dimensions
    n_profiles = len(profiles)
    rows = (n_profiles + cols - 1) // cols  # Ceiling division

    # Set figure size
    if figsize is None:
        figsize = (7 * cols, 3.5 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()  # Ensure 1D array

    # Plot each profile
    for i, prof in enumerate(profiles):
        ax = axes[i]

        # Apply smoothing
        elevations = gaussian_filter1d(prof.elevations, sigma=smooth) if smooth else prof.elevations.copy()
        distances = prof.distances.copy()

        # Optionally ensure profile descends from higher to lower elevation
        if ensure_descending:
            distances, elevations = Profile._ensure_descending(distances, elevations)

        # Normalize if requested
        if normalize:
            # Normalize elevation to 0-1
            elev_min, elev_max = float(elevations.min()), float(elevations.max())
            if elev_max > elev_min:
                elevations = (elevations - elev_min) / (elev_max - elev_min)

            # Normalize distance to 0-1
            dist_min, dist_max = float(distances.min()), float(distances.max())
            if dist_max > dist_min:
                distances = (distances - dist_min) / (dist_max - dist_min)

        # Set axis limits to exact data range (no padding)
        ylim = (float(elevations.min()), float(elevations.max()))
        xlim = (float(distances.min()), float(distances.max()))

        ax.plot(distances, elevations, **kwargs)
        ax.fill_between(distances, elevations, elevations.min(), alpha=0.3)

        # If main profile provided, mark intersection with vertical line
        if main_profile is not None:
            # For cross-sections, the main profile intersects at the midpoint
            mid_distance = distances[len(distances) // 2]
            ax.axvline(mid_distance, color="black", linestyle="-", linewidth=1.5, alpha=0.7, zorder=10)

        ax.set_xlabel("Normalized distance" if normalize else "Distance (km)")
        ax.set_ylabel("Normalized elevation" if normalize else "Elevation (m)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        title = prof.name if prof.name else f"Profile {i + 1}"
        ax.set_title(f"{title} ({prof.distances[-1]:.1f} km)")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_profiles, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig, axes


def plot_profiles_map(
    profiles: Profile | list[Profile], bathymetry_data=None, main_profile: Profile | None = None, cmap=cmo.deep_r, **kwargs
):
    """
    Plot profile locations on a map.

    Parameters
    ----------
    profiles : Profile or list[Profile]
        Profile object(s) to show on map
    bathymetry_data : xr.DataArray, optional
        Bathymetry data to plot as background. If None, uses data from first profile.
    main_profile : Profile, optional
        Optional main profile to highlight (e.g., when showing cross-sections along a main profile)
    **kwargs
        Additional arguments passed to matplotlib plot()

    Returns
    -------
    Figure, Axes
        Matplotlib figure and axes

    Examples
    --------
    >>> from bathy.profile import plot_profiles_map
    >>> prof1 = bath.profile(-8, 52, -2, 58, name="Profile 1")
    >>> prof2 = bath.profile(-8, 53, -2, 59, name="Profile 2")
    >>> plot_profiles_map([prof1, prof2])
    >>> # Single profile
    >>> plot_profiles_map(prof1)
    >>> # With cross-sections
    >>> main = bath.profile(-9.5, 52.0, -5.5, 54.0)
    >>> sections = Profile.cross_sections(bath.data, main, interval_km=10, section_width_km=20)
    >>> plot_profiles_map(sections, main_profile=main)
    """
    # Handle single profile
    if isinstance(profiles, Profile):
        profiles = [profiles]

    if not profiles:
        raise ValueError("Need at least one profile to plot")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot bathymetry background
    # Use provided data, or get from first profile if not provided
    if bathymetry_data is None and profiles:
        bathymetry_data = profiles[0].data

    if bathymetry_data is not None:
        extent = get_extent(bathymetry_data)
        ax.imshow(bathymetry_data.values, cmap=cmap, origin="lower", extent=extent, aspect="auto", alpha=0.6)

    # Plot each profile line
    for i, prof in enumerate(profiles, start=1):
        label = prof.name if prof.name else f"Profile {i}"

        # Check if profile has custom path (from shapefile/coordinates)
        if "path_lons" in prof.metadata and "path_lats" in prof.metadata:
            lons = prof.metadata["path_lons"]
            lats = prof.metadata["path_lats"]
            ax.plot(lons, lats, "-", linewidth=2, label=label, **kwargs)
            ax.plot(lons[0], lats[0], "o", markersize=8)
            ax.plot(lons[-1], lats[-1], "s", markersize=8)
        else:
            # Simple straight line profile
            ax.plot([prof.start_lon, prof.end_lon], [prof.start_lat, prof.end_lat], "-", linewidth=2, label=label, **kwargs)
            ax.plot(prof.start_lon, prof.start_lat, "o", markersize=8)
            ax.plot(prof.end_lon, prof.end_lat, "s", markersize=8)

    # Plot main profile if provided (in a distinctive style)
    if main_profile is not None:
        main_label = main_profile.name if main_profile.name else "Main Profile"
        ax.plot(
            [main_profile.start_lon, main_profile.end_lon],
            [main_profile.start_lat, main_profile.end_lat],
            "r-",
            linewidth=3,
            label=main_label,
            zorder=10,
        )
        ax.plot(main_profile.start_lon, main_profile.start_lat, "go", markersize=10, zorder=11, label="Start")
        ax.plot(main_profile.end_lon, main_profile.end_lat, "rs", markersize=10, zorder=11, label="End")

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    if profiles and profiles[0].name or main_profile is not None:
        ax.legend()

    return fig, ax
