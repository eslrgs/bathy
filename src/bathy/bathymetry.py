"""Bathymetry class with loading, analysis, and visualisation."""

import os
import tempfile
from typing import TYPE_CHECKING
from urllib.parse import urlencode
from urllib.request import urlretrieve

import cmocean.cm as cmo
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import rioxarray
import xarray as xr
from geographiclib.geodesic import Geodesic
from matplotlib.colors import BoundaryNorm, ListedColormap
from xrspatial import hillshade

from bathy.utils import get_extent

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from bathy.profile import Profile


# Preset regions dictionary: {name: (lon_min, lon_max, lat_min, lat_max)}
REGIONS = {
    # Atlantic Ocean
    "north_atlantic": (-80, 0, 40, 70),
    "mid_atlantic_ridge": (-45, -15, -30, 30),
    "gulf_of_mexico": (-98, -80, 18, 31),
    "caribbean": (-90, -60, 10, 25),
    # Pacific Ocean
    "north_pacific": (140, -120, 30, 60),
    "mariana_trench": (140, 148, 10, 15),
    "east_pacific_rise": (-115, -105, -20, 20),
    "galapagos": (-92, -88, -2, 2),
    # Indian Ocean
    "arabian_sea": (55, 75, 10, 25),
    "bay_of_bengal": (80, 95, 5, 22),
    "indian_ocean_ridge": (60, 80, -30, 0),
    # Mediterranean & European Seas
    "mediterranean": (-6, 37, 30, 46),
    "black_sea": (27, 42, 41, 47),
    "north_sea": (-4, 9, 51, 62),
    "baltic_sea": (10, 30, 53, 66),
    # Arctic & Antarctic
    "arctic": (-180, 180, 70, 90),
    "greenland": (-55, -20, 60, 83),
    "antarctic": (-180, 180, -90, -60),
    "ross_sea": (160, -140, -78, -70),
    # Southeast Asia
    "south_china_sea": (105, 120, 5, 23),
    "philippine_trench": (125, 130, 8, 12),
    "java_trench": (105, 120, -12, -8),
    # Regional Seas
    "red_sea": (32, 44, 12, 30),
    "persian_gulf": (48, 57, 24, 30),
    "coral_sea": (145, 160, -25, -10),
    "tasman_sea": (150, 165, -45, -30),
    # Ridges & Features
    "east_pacific_rise_full": (-115, -105, -55, 55),
    "southwest_indian_ridge": (20, 70, -50, -25),
    # Continental Margins
    "us_east_coast": (-78, -65, 30, 45),
    "us_west_coast": (-130, -115, 30, 50),
    "europe_west_coast": (-12, 0, 40, 60),
    "japan_trench": (140, 148, 30, 45),
}


def list_regions() -> list[str]:
    """
    List all available preset regions.

    Returns
    -------
    list[str]
        Sorted list of region names

    Examples
    --------
    >>> from bathy import list_regions
    >>> regions = list_regions()
    >>> print(regions[:5])
    ['antarctic', 'arabian_sea', 'arctic', 'baltic_sea', 'bay_of_bengal']
    """
    return sorted(REGIONS.keys())


def _get_region(name: str) -> tuple[float, float, float, float]:
    """
    Get coordinates for a preset region.

    Parameters
    ----------
    name : str
        Region name (see list_regions())

    Returns
    -------
    tuple[float, float, float, float]
        (lon_min, lon_max, lat_min, lat_max)
    """
    if name not in REGIONS:
        available = ", ".join(list_regions()[:5])
        raise ValueError(f"Region '{name}' not found. Available: {available}, ... (see bathy.list_regions())")
    return REGIONS[name]


class Bathymetry:
    """
    Bathymetry data with analysis and visualisation methods.

    Parameters
    ----------
    filepath : str
        Path to the NetCDF file
    lon_range : tuple[float, float], optional
        Longitude bounds (min, max). Cannot be used with 'region'.
    lat_range : tuple[float, float], optional
        Latitude bounds (min, max). Cannot be used with 'region'.
    region : str, optional
        Preset region name (e.g., 'mediterranean', 'mariana_trench').
        See `bathy.list_regions()` for available regions.
        Cannot be used with 'lon_range' or 'lat_range'.
    var_name : str, default 'elevation'
        Variable name
    lon_name : str, default 'lon'
        Longitude coordinate name
    lat_name : str, default 'lat'
        Latitude coordinate name

    Attributes
    ----------
    data : xr.DataArray
        The elevation data
    filepath : str
        Path to source file

    Examples
    --------
    >>> # Using explicit coordinates
    >>> bath = Bathymetry('gebco.nc', lon_range=(-10, -5), lat_range=(50, 55))
    >>> # Using preset region
    >>> bath = Bathymetry('gebco.nc', region='mediterranean')
    >>> bath.summary()
    >>> bath.plot_bathy()
    """

    def __init__(
        self,
        filepath: str,
        lon_range: tuple[float, float] | None = None,
        lat_range: tuple[float, float] | None = None,
        region: str | None = None,
        var_name: str = "elevation",
        lon_name: str = "lon",
        lat_name: str = "lat",
    ):
        self.filepath = filepath

        # Handle region parameter
        if region is not None:
            if lon_range is not None or lat_range is not None:
                raise ValueError("Cannot specify both 'region' and 'lon_range'/'lat_range'. Use one or the other.")
            lon_min, lon_max, lat_min, lat_max = _get_region(region)
            lon_range = (lon_min, lon_max)
            lat_range = (lat_min, lat_max)

        # Validate range inputs
        if lon_range is not None and len(lon_range) != 2:
            raise ValueError(f"lon_range must be a 2-tuple, got {len(lon_range)} elements")
        if lat_range is not None and len(lat_range) != 2:
            raise ValueError(f"lat_range must be a 2-tuple, got {len(lat_range)} elements")

        # Load data using internal method
        self.data = self._load_data(lon_range, lat_range, var_name, lon_name, lat_name)

    @property
    def lon_range(self) -> tuple[float, float]:
        """Longitude range (min, max)."""
        return (float(self.data.lon.min()), float(self.data.lon.max()))

    @property
    def lat_range(self) -> tuple[float, float]:
        """Latitude range (min, max)."""
        return (float(self.data.lat.min()), float(self.data.lat.max()))

    @property
    def shape(self) -> tuple[int, ...]:
        """Data shape."""
        return self.data.shape

    @classmethod
    def from_gebco_opendap(
        cls,
        lon_range: tuple[float, float] | None = None,
        lat_range: tuple[float, float] | None = None,
        region: str | None = None,
        year: int = 2025,
        save_path: str | None = None,
    ) -> "Bathymetry":
        """
        Download GEBCO data from OPeNDAP server for a specific region.

        This downloads only the requested region without downloading the entire dataset.

        Parameters
        ----------
        lon_range : tuple[float, float], optional
            Longitude bounds (min, max), range: -180 to 180. Cannot be used with 'region'.
        lat_range : tuple[float, float], optional
            Latitude bounds (min, max), range: -90 to 90. Cannot be used with 'region'.
        region : str, optional
            Preset region name (e.g., 'mediterranean', 'mariana_trench').
            See `bathy.list_regions()` for available regions.
            Cannot be used with 'lon_range' or 'lat_range'.
        year : int, default 2025
            GEBCO dataset year
        save_path : str, optional
            If provided, save the downloaded data to this path

        Returns
        -------
        Bathymetry
            Bathymetry object with the downloaded data

        Examples
        --------
        Download a region using coordinates:

        >>> bath = Bathymetry.from_gebco_opendap(
        ...     lon_range=(-10, -5),
        ...     lat_range=(50, 55)
        ... )

        Download a preset region:

        >>> bath = Bathymetry.from_gebco_opendap(region='mediterranean')

        Download and save to file:

        >>> bath = Bathymetry.from_gebco_opendap(
        ...     region='mediterranean',
        ...     save_path='mediterranean.nc'
        ... )
        """
        # Handle region parameter
        if region is not None:
            if lon_range is not None or lat_range is not None:
                raise ValueError("Cannot specify both 'region' and 'lon_range'/'lat_range'. Use one or the other.")
            lon_min, lon_max, lat_min, lat_max = _get_region(region)
            lon_range = (lon_min, lon_max)
            lat_range = (lat_min, lat_max)

        # Validate that we have ranges
        if lon_range is None or lat_range is None:
            raise ValueError("Must specify either 'region' or both 'lon_range' and 'lat_range'")

        filepath = cls._download_gebco(lon_range, lat_range, year, save_path)
        return cls(filepath, var_name="elevation", lon_name="lon", lat_name="lat")

    @classmethod
    def from_array(cls, data: xr.DataArray) -> "Bathymetry":
        """
        Create a Bathymetry object directly from an xarray DataArray.

        Parameters
        ----------
        data : xr.DataArray
            Elevation data with 'lon' and 'lat' coordinates

        Returns
        -------
        Bathymetry
            Bathymetry object with the provided data

        Examples
        --------
        >>> import xarray as xr
        >>> import numpy as np
        >>> data = xr.DataArray(
        ...     np.random.rand(10, 10) * -100,
        ...     coords={"lon": np.linspace(-10, -5, 10), "lat": np.linspace(50, 55, 10)},
        ...     dims=["lat", "lon"],
        ... )
        >>> bath = Bathymetry.from_array(data)
        """
        obj = cls.__new__(cls)
        obj.filepath = None
        obj.data = data
        return obj

    @staticmethod
    def _download_gebco(
        lon_range: tuple[float, float],
        lat_range: tuple[float, float],
        year: int,
        save_path: str | None,
    ) -> str:
        """Download GEBCO data from THREDDS server."""
        # Use THREDDS NetCDF Subset Service (NCSS) for fast server-side subsetting
        base_url = f"https://dap.ceda.ac.uk/thredds/ncss/bodc/gebco/global/gebco_{year}/ice_surface_elevation/netcdf/GEBCO_{year}.nc"

        # Build query parameters for spatial subset
        params = {
            "var": "elevation",
            "north": max(lat_range),
            "south": min(lat_range),
            "west": min(lon_range),
            "east": max(lon_range),
        }

        ncss_url = f"{base_url}?{urlencode(params)}"

        # Download the subset directly from server
        if save_path:
            filepath = save_path
        else:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nc")
            filepath = temp_file.name

        urlretrieve(ncss_url, filepath)
        return filepath

    def to_geotiff(
        self,
        filepath: str,
        crs: str = "EPSG:4326",
        **kwargs,
    ) -> None:
        """
        Save bathymetry data to a GeoTIFF file using rioxarray.

        Parameters
        ----------
        filepath : str
            Output GeoTIFF file path
        crs : str, default 'EPSG:4326'
            Coordinate reference system (e.g., 'EPSG:4326' for WGS84)
        **kwargs
            Additional arguments passed to rioxarray.to_raster()

        Examples
        --------
        Save bathymetry to GeoTIFF:

        >>> bath.to_geotiff('output.tif')

        Save with a different CRS:

        >>> bath.to_geotiff('output.tif', crs='EPSG:3857')
        """
        # Set the CRS if not already set
        if not hasattr(self.data, "rio") or self.data.rio.crs is None:
            self.data = self.data.rio.write_crs(crs)

        self.data.rio.to_raster(filepath, **kwargs)

    # Internal utilities

    def _load_data(
        self,
        lon_range: tuple[float, float] | None,
        lat_range: tuple[float, float] | None,
        var_name: str,
        lon_name: str,
        lat_name: str,
    ) -> xr.DataArray:
        """Load data from file based on file type."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        # Detect file type and load appropriately
        ext = os.path.splitext(self.filepath)[1].lower()
        if ext in [".tif", ".tiff"]:
            return self._load_geotiff()
        return self._load_netcdf(lon_range, lat_range, var_name, lon_name, lat_name)

    def _load_geotiff(self) -> xr.DataArray:
        """Load GeoTIFF file with rioxarray."""
        da = rioxarray.open_rasterio(self.filepath, masked=True)

        # Select first band if multi-band
        if "band" in da.dims:
            da = da.sel(band=1)

        # Rename spatial coordinates to lon/lat
        if "x" in da.dims and "y" in da.dims:
            da = da.rename({"x": "lon", "y": "lat"})

        return da

    def _load_netcdf(
        self,
        lon_range: tuple[float, float] | None,
        lat_range: tuple[float, float] | None,
        var_name: str,
        lon_name: str,
        lat_name: str,
    ) -> xr.DataArray:
        """Load NetCDF file with xarray."""
        ds = xr.open_dataset(self.filepath)

        # Check if specified names exist
        if lon_name not in ds.coords and lon_name not in ds.dims:
            raise ValueError(f"Longitude coordinate '{lon_name}' not found. Available: {list(ds.coords)}")
        if lat_name not in ds.coords and lat_name not in ds.dims:
            raise ValueError(f"Latitude coordinate '{lat_name}' not found. Available: {list(ds.coords)}")
        if var_name not in ds.data_vars:
            raise ValueError(f"Variable '{var_name}' not found. Available: {list(ds.data_vars)}")

        # Store original bounds before selection for error messages
        original_lon_bounds = (float(ds[lon_name].min()), float(ds[lon_name].max()))
        original_lat_bounds = (float(ds[lat_name].min()), float(ds[lat_name].max()))

        # Apply range selections
        if lon_range:
            ds = ds.sel({lon_name: slice(*lon_range)})
        if lat_range:
            ds = ds.sel({lat_name: slice(*sorted(lat_range))})

        data = ds[var_name]

        # Rename coordinates to standard names for internal consistency
        rename_dict = {}
        if lon_name != "lon":
            rename_dict[lon_name] = "lon"
        if lat_name != "lat":
            rename_dict[lat_name] = "lat"
        if rename_dict:
            data = data.rename(rename_dict)

        # Validate that data is not empty after selection
        if data.size == 0:
            raise ValueError(
                f"Data selection resulted in empty array. "
                f"Requested: lon={lon_range}, lat={lat_range}. "
                f"Available: lon={original_lon_bounds}, lat={original_lat_bounds}"
            )

        return data

    def _add_contours(self, ax: "Axes", contours: int | list[float], **kwargs) -> None:
        """Add contour lines to an existing axes."""
        cs = self.data.plot.contour(ax=ax, levels=contours, colors="black", alpha=0.8, linewidths=1, linestyles="-", **kwargs)
        ax.clabel(cs, inline=True, fontsize=8)

    @staticmethod
    def _clean_values(data: xr.DataArray) -> np.ndarray:
        """Get flattened array with NaN values removed."""
        values = data.values.ravel()
        return values[~np.isnan(values)]

    # Analysis methods

    def summary(self) -> pl.DataFrame:
        """
        Generate summary statistics.

        Returns
        -------
        pl.DataFrame
            DataFrame with statistics (min, max, mean, median, std, q25, q75)
        """
        values = self._clean_values(self.data)

        return pl.DataFrame(
            {
                "statistic": ["min", "max", "mean", "median", "std", "q25", "q75"],
                "value": [
                    float(np.min(values)),
                    float(np.max(values)),
                    float(np.mean(values)),
                    float(np.median(values)),
                    float(np.std(values)),
                    float(np.percentile(values, 25)),
                    float(np.percentile(values, 75)),
                ],
            }
        )

    def depth_stats(self) -> pl.DataFrame:
        """
        Statistics for underwater areas only.

        Returns
        -------
        pl.DataFrame
            DataFrame with depth statistics
        """
        underwater = self.data.values[self.data.values < 0]

        if not len(underwater):
            return pl.DataFrame(
                {
                    "statistic": ["min_depth", "max_depth", "mean_depth", "median_depth", "std_depth"],
                    "value": [np.nan, np.nan, np.nan, np.nan, np.nan],
                }
            )

        return pl.DataFrame(
            {
                "statistic": ["min_depth", "max_depth", "mean_depth", "median_depth", "std_depth"],
                "value": [
                    float(np.min(underwater)),
                    float(np.max(underwater)),
                    float(np.mean(underwater)),
                    float(np.median(underwater)),
                    float(np.std(underwater)),
                ],
            }
        )

    def coverage(self) -> pl.DataFrame:
        """
        Calculate land/sea coverage.

        Returns
        -------
        pl.DataFrame
            DataFrame with coverage statistics
        """
        total = self.data.size
        underwater = np.sum(self.data.values < 0)
        land = np.sum(self.data.values >= 0)

        return pl.DataFrame(
            {
                "metric": ["total_cells", "underwater_cells", "land_cells", "underwater_pct", "land_pct"],
                "value": [
                    float(total),
                    float(underwater),
                    float(land),
                    float(underwater / total * 100),
                    float(land / total * 100),
                ],
            }
        )

    def hypsometric_index(self) -> float:
        """
        Calculate the hypsometric index (HI).

        The hypsometric index quantifies the distribution of elevation within a
        region as a single value between 0 and 1:

        HI = (mean - min) / (max - min)

        Returns
        -------
        float
            Hypsometric index where:
            - HI > 0.5: Convex hypsometry (more area at higher elevations)
            - HI ≈ 0.5: Equilibrium (S-shaped distribution)
            - HI < 0.5: Concave hypsometry (more area at lower elevations)

        Examples
        --------
        >>> bath = Bathymetry.from_gebco_opendap(region='mediterranean')
        >>> hi = bath.hypsometric_index()
        >>> print(f"Hypsometric Index: {hi:.3f}")
        """
        values = self._clean_values(self.data)
        if len(values) == 0:
            return np.nan
        h_mean = np.mean(values)
        h_min = np.min(values)
        h_max = np.max(values)
        if h_max == h_min:
            return np.nan
        return float((h_mean - h_min) / (h_max - h_min))

    def hypsometric_curve(self, bins: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the hypsometric curve.

        The hypsometric curve shows the cumulative distribution of area with
        elevation, normalised to relative values (0 to 1).

        Parameters
        ----------
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
        >>> bath = Bathymetry.from_gebco_opendap(region='mediterranean')
        >>> rel_area, rel_elev = bath.hypsometric_curve()
        >>> plt.plot(rel_area, rel_elev)
        >>> plt.xlabel('Relative Area (a/A)')
        >>> plt.ylabel('Relative Elevation (h/H)')
        """
        values = self._clean_values(self.data)
        h_min, h_max = values.min(), values.max()

        bin_edges = np.linspace(h_min, h_max, bins + 1)
        counts, _ = np.histogram(values, bins=bin_edges)

        # Cumulative area above each elevation
        cumulative = np.cumsum(counts[::-1])[::-1]
        relative_area = cumulative / cumulative[0]

        # Normalised elevation (bin centres)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        relative_elevation = (bin_centres - h_min) / (h_max - h_min)

        return relative_area, relative_elevation

    def plot_hypsometric_curve(self, bins: int = 100, **kwargs) -> None:
        """
        Plot the hypsometric curve.

        Parameters
        ----------
        bins : int, default 100
            Number of elevation bins
        **kwargs
            Additional arguments passed to plt.plot

        Examples
        --------
        >>> bath = Bathymetry.from_gebco_opendap(region='mediterranean')
        >>> bath.plot_hypsometric_curve()
        """
        rel_area, rel_elev = self.hypsometric_curve(bins)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(rel_area, rel_elev, linewidth=2, **kwargs)
        ax.plot([0, 1], [1, 0], "k--", alpha=0.3)
        ax.set_xlabel("Relative Area (a/A)")
        ax.set_ylabel("Relative Elevation (h/H)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        plt.show()

    def _cell_size_metres(self) -> tuple[float, float]:
        """Return (dy, dx) cell size in metres using geodesic measurement."""
        lat_spacing = np.abs(np.diff(self.data.lat.values).mean())
        lon_spacing = np.abs(np.diff(self.data.lon.values).mean())
        lat_centre = float(self.data.lat.mean())
        lon_centre = float(self.data.lon.mean())

        geod = Geodesic.WGS84
        dy = geod.Inverse(lat_centre, lon_centre, lat_centre + lat_spacing, lon_centre)["s12"]
        dx = geod.Inverse(lat_centre, lon_centre, lat_centre, lon_centre + lon_spacing)["s12"]
        return dy, dx

    def slope(self) -> xr.DataArray:
        """
        Calculate seafloor slope in degrees.

        Returns
        -------
        xr.DataArray
            Slope magnitude in degrees
        """
        dy, dx = self._cell_size_metres()
        gy, gx = np.gradient(self.data.values, dy, dx)
        slope_deg = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
        return xr.DataArray(slope_deg, coords=self.data.coords, dims=self.data.dims, name="slope")

    def curvature(self) -> xr.DataArray:
        """
        Calculate seafloor curvature (Laplacian).

        Returns
        -------
        xr.DataArray
            Curvature values (positive = convex/ridges, negative = concave/valleys)
        """
        dy, dx = self._cell_size_metres()
        gy, gx = np.gradient(self.data.values, dy, dx)
        gyy, _ = np.gradient(gy, dy, dx)
        _, gxx = np.gradient(gx, dy, dx)
        return xr.DataArray(gxx + gyy, coords=self.data.coords, dims=self.data.dims, name="curvature")

    # Profile and Swath methods

    def profile(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        num_points: int | None = None,
        point_spacing: float | None = None,
        name: str | None = None,
    ) -> "Profile":
        """
        Create a bathymetric profile.

        Parameters
        ----------
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

        Returns
        -------
        Profile
            Profile object for analysis

        Examples
        --------
        Create a profile with default 100 points:

        >>> prof = bath.profile(start=(-9.5, 52.0), end=(-5.5, 52.0))

        Create a profile with a point every 1 km:

        >>> prof = bath.profile(start=(-9.5, 52.0), end=(-5.5, 52.0), point_spacing=1.0)
        """
        from bathy.profile import Profile  # noqa: PLC0415

        start_lon, start_lat = start
        end_lon, end_lat = end
        return Profile(self.data, start_lon, start_lat, end_lon, end_lat, num_points, point_spacing, name)

    # Plotting methods

    def plot_bathy(self, contours: int | list[float] | None = None, cmap=None, **kwargs) -> None:
        """
        Plot bathymetry elevation.

        Parameters
        ----------
        contours : int or list[float], optional
            If int, number of contour levels to plot
            If list, specific contour levels (in meters)
            If None, no contours are plotted
        cmap : str or Colormap, optional
            Colormap to use. Defaults to cmocean 'deep_r' (reversed, perceptually uniform,
            colorblind-friendly bathymetry colormap with light=shallow, dark=deep)
        **kwargs
            Additional arguments passed to xarray plot
        """
        if cmap is None:
            cmap = cmo.deep_r  # Reversed so light=shallow, dark=deep

        fig, ax = plt.subplots(figsize=(10, 8))

        # Mask land (elevation >= 0)
        data_masked = self.data.where(self.data < 0)

        # Set colorbar label if not provided
        if "cbar_kwargs" not in kwargs:
            kwargs["cbar_kwargs"] = {"label": "Elevation (m)"}

        data_masked.plot(ax=ax, cmap=cmap, **kwargs)

        if contours is not None:
            self._add_contours(ax, contours)

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        plt.show()

    def plot_hillshade(self, azimuth: float = 315, altitude: float = 45, contours: int | list[float] | None = None, **kwargs) -> None:
        """
        Create hillshade visualisation.

        Parameters
        ----------
        azimuth : float, default 315
            Light source azimuth in degrees
        altitude : float, default 45
            Light source altitude in degrees
        contours : int or list[float], optional
            If int, number of contour levels to plot
            If list, specific contour levels (in meters)
            If None, no contours are plotted
        **kwargs
            Additional arguments passed to imshow
        """
        shaded = hillshade(-self.data, azimuth=azimuth, angle_altitude=altitude)

        extent = get_extent(self.data)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(shaded, cmap="gray", origin="lower", extent=extent, aspect="auto", **kwargs)

        if contours is not None:
            self._add_contours(ax, contours)

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        plt.show()

    def plot_slope(self, contours: int | list[float] | None = None, vmax: float | None = None, **kwargs) -> None:
        """
        Plot seafloor slope.

        Parameters
        ----------
        contours : int or list[float], optional
            If int, number of contour levels to plot
            If list, specific contour levels (in meters)
            If None, no contours are plotted
        vmax : float, optional
            Maximum slope value for colour scale. Useful for clipping outliers.
            Default uses the 99th percentile to avoid extreme values dominating.
        **kwargs
            Additional arguments passed to imshow
        """
        slope_data = self.slope()
        extent = get_extent(self.data)

        # Use 99th percentile as default vmax to handle outliers
        if vmax is None:
            vmax = float(np.nanpercentile(slope_data.values, 99))

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            slope_data.values,
            cmap="Greys",  # Grey scale: white=flat, dark grey=steep
            origin="lower",
            extent=extent,
            aspect="auto",
            vmin=0,
            vmax=vmax,
            **kwargs,
        )
        plt.colorbar(im, ax=ax, label="Slope (°)")

        if contours is not None:
            self._add_contours(ax, contours)

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        plt.show()

    def plot_curvature(self, contours: int | list[float] | None = None, **kwargs) -> None:
        """
        Plot seafloor curvature.

        Visualises the rate of change of slope to identify convex features (ridges,
        seamounts) and concave features (valleys, trenches, canyons).

        Parameters
        ----------
        contours : int or list[float], optional
            If int, number of contour levels to plot
            If list, specific contour levels (in metres)
            If None, no contours are plotted
        **kwargs
            Additional arguments passed to imshow

        Notes
        -----
        The diverging colourmap centres on zero curvature (flat surfaces):
        - Red/warm colours indicate positive curvature (convex features like ridges)
        - Blue/cool colours indicate negative curvature (concave features like valleys)
        """
        curvature_data = self.curvature()
        extent = get_extent(self.data)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Use a diverging colormap centred on zero
        vmax = np.nanmax(np.abs(curvature_data.values))
        im = ax.imshow(
            curvature_data.values,
            cmap=cmo.balance,
            origin="lower",
            extent=extent,
            aspect="auto",
            vmin=-vmax,
            vmax=vmax,
            **kwargs,
        )
        plt.colorbar(im, ax=ax, label="Curvature")

        if contours is not None:
            self._add_contours(ax, contours)

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        plt.show()

    def plot_histogram(self, bins: int = 50, **kwargs) -> None:
        """Plot histogram of elevation values."""
        fig, ax = plt.subplots(figsize=(10, 6))

        values = self._clean_values(self.data)

        ax.hist(values, bins=bins, edgecolor="black", **kwargs)
        ax.axvline(0, color="blue", linestyle="--", linewidth=2, label="Sea level")
        ax.set_xlabel("Elevation (m)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()

    def plot_depth_zones(
        self,
        zones: list[float] | None = None,
        labels: list[str] | None = None,
        contours: int | list[float] | None = None,
        **kwargs,
    ) -> None:
        """
        Plot bathymetry color-coded by depth zones.

        Parameters
        ----------
        zones : list[float], optional
            Depth boundaries (default: [0, -200, -1000, -4000])
        labels : list[str], optional
            Zone labels (default: ['Shelf', 'Slope', 'Abyss', 'Deep'])
        contours : int or list[float], optional
            If int, number of contour levels to plot
            If list, specific contour levels (in meters)
            If None, no contours are plotted
        **kwargs
            Additional arguments passed to imshow
        """
        if zones is None:
            zones = [0, -200, -1000, -4000]

        if labels is None:
            labels = ["Shelf", "Slope", "Abyss", "Deep"]

        # Sort zones in ascending order (most negative to 0) for BoundaryNorm
        sorted_zones = sorted(zones)
        n_zones = len(sorted_zones)

        # Add the minimum data value as the lower bound
        boundaries = [self.data.min().values] + sorted_zones

        # Reverse labels to match boundary order (deepest to shallowest)
        reversed_labels = labels[::-1]

        # Create discrete colormap from cmocean deep (dark=deep, light=shallow)
        # Reverse colors so deepest zone (first boundary) gets darkest color
        deep_colors = cmo.deep(np.linspace(1, 0, n_zones))
        colors = ListedColormap(deep_colors)
        norm = BoundaryNorm(boundaries, n_zones)

        extent = get_extent(self.data)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(self.data.values, cmap=colors, norm=norm, origin="lower", extent=extent, aspect="auto", **kwargs)

        if contours is not None:
            self._add_contours(ax, contours)

        cbar = plt.colorbar(im, ax=ax, label="Depth zone")

        # Set ticks at the center of each color band with depth range labels
        tick_positions = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(n_zones)]
        tick_labels = [f"{reversed_labels[i]}\n({int(boundaries[i + 1])} to {int(boundaries[i])} m)" for i in range(n_zones)]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        plt.show()

    def plot_surface3d(
        self,
        stride: int = 10,
        vertical_exaggeration: float = 50.0,
        smooth: int | None = None,
        elev: float = 30,
        azim: float = -60,
        **kwargs,
    ) -> None:
        """
        Create static 3D surface plot.

        Parameters
        ----------
        stride : int, default 10
            Stride for downsampling the data (every Nth point)
        vertical_exaggeration : float, default 50.0
            Factor to exaggerate the vertical scale for better visualisation.
        smooth : int, optional
            Apply a uniform filter with this kernel size to smooth the surface.
            Typical values are 3-7.
        elev : float, default 30
            Elevation viewing angle in degrees. 0 is horizontal, 90 is directly above.
        azim : float, default -60
            Azimuth viewing angle in degrees. Rotates around the vertical axis.
        **kwargs
            Additional arguments passed to plot_surface
        """
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Downsample data
        lon = self.data.lon.values[::stride]
        lat = self.data.lat.values[::stride]
        z = self.data.values[::stride, ::stride]

        # Apply smoothing if requested
        if smooth is not None:
            from scipy.ndimage import uniform_filter  # noqa: PLC0415

            z = uniform_filter(z, size=smooth, mode="nearest")

        lon_grid, lat_grid = np.meshgrid(lon, lat)

        surf = ax.plot_surface(lon_grid, lat_grid, z, cmap=cmo.deep_r, linewidth=0, antialiased=True, **kwargs)
        fig.colorbar(surf, ax=ax, label="Elevation (m)", shrink=0.5, pad=0.1)

        # Set aspect ratio accounting for longitude compression at higher latitudes
        lat_centre = float(self.data.lat.mean())
        lon_scale = np.cos(np.radians(lat_centre))
        ax.set_box_aspect([np.ptp(lon) * lon_scale, np.ptp(lat), np.ptp(z) * vertical_exaggeration / 1000])

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_zlabel("Elevation (m)")
        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        """String representation."""
        lon_min, lon_max = self.lon_range
        lat_min, lat_max = self.lat_range
        return f"Bathymetry(shape={self.shape}, lon=[{lon_min:.2f}, {lon_max:.2f}], lat=[{lat_min:.2f}, {lat_max:.2f}])"
