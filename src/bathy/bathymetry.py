"""Bathymetry class with loading, analysis, and visualisation."""

import logging
import os
import tempfile
from typing import TYPE_CHECKING
from urllib.parse import urlencode
from urllib.request import urlopen

import cmocean.cm as cmo
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import rioxarray
import xarray as xr
from geographiclib.geodesic import Geodesic
from matplotlib.colors import BoundaryNorm, ListedColormap
from tqdm import tqdm
from xrspatial import hillshade

from bathy.utils import get_extent

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from bathy.profile import Profile

logger = logging.getLogger(__name__)

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
        raise ValueError(
            f"Region '{name}' not found. "
            f"Available: {available}, ... "
            f"(see bathy.list_regions())"
        )
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
                raise ValueError(
                    "Cannot specify both 'region' and 'lon_range'/'lat_range'."
                )
            lon_min, lon_max, lat_min, lat_max = _get_region(region)
            lon_range = (lon_min, lon_max)
            lat_range = (lat_min, lat_max)

        # Validate range inputs
        if lon_range is not None and len(lon_range) != 2:
            raise ValueError(
                f"lon_range must be a 2-tuple, got {len(lon_range)} elements"
            )
        if lat_range is not None and len(lat_range) != 2:
            raise ValueError(
                f"lat_range must be a 2-tuple, got {len(lat_range)} elements"
            )

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
            Longitude bounds (min, max), range: -180 to 180.
            Cannot be used with 'region'.
        lat_range : tuple[float, float], optional
            Latitude bounds (min, max), range: -90 to 90.
            Cannot be used with 'region'.
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
                raise ValueError(
                    "Cannot specify both 'region' and 'lon_range'/'lat_range'."
                )
            lon_min, lon_max, lat_min, lat_max = _get_region(region)
            lon_range = (lon_min, lon_max)
            lat_range = (lat_min, lat_max)

        # Validate that we have ranges
        if lon_range is None or lat_range is None:
            raise ValueError(
                "Must specify either 'region' or both 'lon_range' and 'lat_range'"
            )

        if save_path and os.path.exists(save_path):
            logger.info(f"Using existing file: {save_path}")
            filepath = save_path
        else:
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
        ...     coords={
        ...         "lon": np.linspace(-10, -5, 10),
        ...         "lat": np.linspace(50, 55, 10),
        ...     },
        ...     dims=["lat", "lon"],
        ... )
        >>> bath = Bathymetry.from_array(data)
        """
        obj = cls.__new__(cls)
        obj.filepath = None
        obj.data = data
        return obj

    def clip(
        self,
        lon_range: tuple[float, float] | None = None,
        lat_range: tuple[float, float] | None = None,
    ) -> "Bathymetry":
        """
        Clip to a smaller region, returning a new Bathymetry object.

        Parameters
        ----------
        lon_range : tuple[float, float], optional
            Longitude bounds (min, max)
        lat_range : tuple[float, float], optional
            Latitude bounds (min, max)

        Returns
        -------
        Bathymetry
            New Bathymetry object with clipped data

        Examples
        --------
        >>> subset = bath.clip(lon_range=(-5, 5), lat_range=(35, 40))
        """
        data = self.data
        if lon_range is not None:
            data = data.sel(lon=slice(*lon_range))
        if lat_range is not None:
            data = data.sel(lat=slice(*sorted(lat_range)))
        return Bathymetry.from_array(data)

    @staticmethod
    def _download_gebco(
        lon_range: tuple[float, float],
        lat_range: tuple[float, float],
        year: int,
        save_path: str | None,
    ) -> str:
        """Download GEBCO data from THREDDS server."""
        params = {
            "var": "elevation",
            "north": max(lat_range),
            "south": min(lat_range),
            "west": min(lon_range),
            "east": max(lon_range),
        }

        base_url = f"https://dap.ceda.ac.uk/thredds/ncss/bodc/gebco/global/gebco_{year}/ice_surface_elevation/netcdf/GEBCO_{year}.nc"
        ncss_url = f"{base_url}?{urlencode(params)}"

        filepath = (
            save_path or tempfile.NamedTemporaryFile(delete=False, suffix=".nc").name
        )

        logger.info(f"Downloading GEBCO {year} data from CEDA...")

        response = urlopen(ncss_url)
        total = int(response.headers.get("Content-Length", 0))

        with (
            open(filepath, "wb") as f,
            tqdm(
                total=total, unit="B", unit_scale=True, desc="Downloading GEBCO"
            ) as pbar,
        ):
            while chunk := response.read(8192):
                f.write(chunk)
                pbar.update(len(chunk))

        logger.info(f"Saved to {filepath}")

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

    def to_netcdf(self, filepath: str, **kwargs) -> None:
        """
        Save bathymetry data to a NetCDF file.

        Parameters
        ----------
        filepath : str
            Output NetCDF file path
        **kwargs
            Additional arguments passed to xarray.DataArray.to_netcdf()

        Examples
        --------
        >>> bath.to_netcdf('output.nc')
        """
        self.data.to_netcdf(filepath, **kwargs)

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
            raise ValueError(
                f"Longitude coordinate '{lon_name}' not found. "
                f"Available: {list(ds.coords)}"
            )
        if lat_name not in ds.coords and lat_name not in ds.dims:
            raise ValueError(
                f"Latitude coordinate '{lat_name}' not found. "
                f"Available: {list(ds.coords)}"
            )
        if var_name not in ds.data_vars:
            raise ValueError(
                f"Variable '{var_name}' not found. Available: {list(ds.data_vars)}"
            )

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
        cs = self.data.plot.contour(
            ax=ax,
            levels=contours,
            colors="black",
            alpha=0.8,
            linewidths=1,
            linestyles="-",
            **kwargs,
        )
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
                    "statistic": [
                        "min_depth",
                        "max_depth",
                        "mean_depth",
                        "median_depth",
                        "std_depth",
                    ],
                    "value": [np.nan, np.nan, np.nan, np.nan, np.nan],
                }
            )

        return pl.DataFrame(
            {
                "statistic": [
                    "min_depth",
                    "max_depth",
                    "mean_depth",
                    "median_depth",
                    "std_depth",
                ],
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
                "metric": [
                    "total_cells",
                    "underwater_cells",
                    "land_cells",
                    "underwater_pct",
                    "land_pct",
                ],
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
        dy = geod.Inverse(lat_centre, lon_centre, lat_centre + lat_spacing, lon_centre)[
            "s12"
        ]
        dx = geod.Inverse(lat_centre, lon_centre, lat_centre, lon_centre + lon_spacing)[
            "s12"
        ]
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
        return xr.DataArray(
            slope_deg, coords=self.data.coords, dims=self.data.dims, name="slope"
        )

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
        return xr.DataArray(
            gxx + gyy, coords=self.data.coords, dims=self.data.dims, name="curvature"
        )

    def bpi(self, radius_km: float = 1.0) -> xr.DataArray:
        """
        Calculate Bathymetric Position Index (BPI).

        BPI measures the relative position of a point compared to its surroundings.
        Positive values indicate ridges or peaks, negative values indicate valleys
        or depressions, and values near zero indicate flat areas or mid-slopes.

        Parameters
        ----------
        radius_km : float, default 1.0
            Radius of the neighbourhood in kilometres (square window)

        Returns
        -------
        xr.DataArray
            BPI values (positive = ridges, negative = valleys)

        Examples
        --------
        >>> bpi = bath.bpi(radius_km=2.0)
        >>> bath.plot_bpi(radius_km=2.0)
        """
        from scipy.ndimage import uniform_filter  # noqa: PLC0415

        # Convert radius to grid cells
        dy, dx = self._cell_size_metres()
        cell_size = (dy + dx) / 2
        window_size = max(3, int(2 * radius_km * 1000 / cell_size) + 1)

        # Calculate neighbourhood mean using fast uniform filter
        neighbourhood_mean = uniform_filter(
            self.data.values.astype(float), size=window_size, mode="nearest"
        )
        bpi_values = self.data.values - neighbourhood_mean

        return xr.DataArray(
            bpi_values, coords=self.data.coords, dims=self.data.dims, name="bpi"
        )

    def rugosity(self, radius_km: float = 1.0) -> xr.DataArray:
        """
        Calculate Vector Ruggedness Measure (VRM).

        VRM measures terrain complexity by decomposing the surface into unit
        normal vectors and quantifying their dispersion within a neighbourhood.
        Values range from 0 (flat) to 1 (maximally rough).

        Parameters
        ----------
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
        >>> rug = bath.rugosity(radius_km=0.5)
        >>> bath.plot_rugosity()
        """
        from scipy.ndimage import uniform_filter  # noqa: PLC0415

        dy, dx = self._cell_size_metres()
        gy, gx = np.gradient(self.data.values, dy, dx)

        slope = np.arctan(np.sqrt(gx**2 + gy**2))
        aspect = np.arctan2(gy, gx)

        x = np.sin(slope) * np.sin(aspect)
        y = np.sin(slope) * np.cos(aspect)
        z = np.cos(slope)

        cell_size = (dy + dx) / 2
        window_size = max(3, int(2 * radius_km * 1000 / cell_size) + 1)

        x_mean = uniform_filter(x, size=window_size, mode="nearest")
        y_mean = uniform_filter(y, size=window_size, mode="nearest")
        z_mean = uniform_filter(z, size=window_size, mode="nearest")

        vrm = 1.0 - np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)

        return xr.DataArray(
            vrm, coords=self.data.coords, dims=self.data.dims, name="rugosity"
        )

    def aspect(self) -> xr.DataArray:
        """
        Calculate seafloor aspect.

        Aspect is the compass direction of the steepest upslope gradient,
        measured in degrees clockwise from north (0° = north, 90° = east,
        180° = south, 270° = west). Flat areas are returned as NaN.

        Returns
        -------
        xr.DataArray
            Aspect in degrees [0, 360), NaN where slope is zero

        Examples
        --------
        >>> asp = bath.aspect()
        >>> bath.plot_aspect()
        """
        dy, dx = self._cell_size_metres()
        gy, gx = np.gradient(self.data.values, dy, dx)

        asp = (90 - np.degrees(np.arctan2(gy, gx))) % 360
        asp[np.sqrt(gx**2 + gy**2) == 0] = np.nan

        return xr.DataArray(
            asp, coords=self.data.coords, dims=self.data.dims, name="aspect"
        )

    def geomorphons(
        self, lookup_km: float = 2.0, flatness_threshold: float = 1.0
    ) -> xr.DataArray:
        """
        Classify seafloor morphology using geomorphons.

        For each cell, a single neighbour is sampled at ``lookup_km`` distance
        in each of the eight principal directions. The line-of-sight angle to
        that neighbour is compared to ``flatness_threshold``; directions are
        coded as positive (+, neighbour above), negative (-, neighbour below),
        or equal (=). The counts of + and - directions determine the class.

        Each direction uses an independent step count so that the physical
        sampling distance is consistent across all eight azimuths (cardinal
        and diagonal directions look at the same distance, not the same number
        of grid cells).

        Parameters
        ----------
        lookup_km : float, default 2.0
            Lookup distance in kilometres. Larger values capture broader forms.
            Recommend at least 5–10 grid cells; for GEBCO (450 m) use ≥ 2 km.
        flatness_threshold : float, default 1.0
            Angle threshold in degrees below which differences are treated as flat.

        Returns
        -------
        xr.DataArray
            Integer class codes (1–10):
            1=flat, 2=peak, 3=ridge, 4=shoulder, 5=spur,
            6=slope, 7=hollow, 8=footslope, 9=valley, 10=pit

        References
        ----------
        Jasiewicz, J., & Stepinski, T.F. (2013). Geomorphons — a pattern
        recognition approach to classification and mapping of landforms.
        Geomorphology, 182, 147–156.

        Examples
        --------
        >>> geom = bath.geomorphons(lookup_km=2.0)
        >>> bath.plot_geomorphons(lookup_km=2.0)
        """
        dy, dx = self._cell_size_metres()

        z = self.data.values.astype(float)
        ny, nx = z.shape

        # 8 principal directions (row_step, col_step)
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
        p = np.zeros((ny, nx), dtype=np.int8)  # neighbours above centre
        m = np.zeros((ny, nx), dtype=np.int8)  # neighbours below centre

        for drow, dcol in directions:
            # Per-direction step count so physical distance ≈ lookup_km in all azimuths.
            # Diagonal steps cover sqrt(2) × more distance per cell than cardinal ones,
            # so they get proportionally fewer steps.
            step_dist = np.sqrt((drow * dy) ** 2 + (dcol * dx) ** 2)
            n = max(1, round(lookup_km * 1000 / step_dist))
            horiz = n * step_dist
            row_off = drow * n
            col_off = dcol * n
            i0 = max(0, -row_off)
            i1 = ny - max(0, row_off)
            j0 = max(0, -col_off)
            j1 = nx - max(0, col_off)

            dz = (
                z[i0 + row_off : i1 + row_off, j0 + col_off : j1 + col_off]
                - z[i0:i1, j0:j1]
            )
            angle = np.arctan2(dz, horiz)
            p[i0:i1, j0:j1] += (angle > threshold_rad).astype(np.int8)
            m[i0:i1, j0:j1] += (angle < -threshold_rad).astype(np.int8)

        # Classify: p = above count, m = below count (each 0–8)
        geom = np.full((ny, nx), 6, dtype=np.int8)  # default: slope
        geom[(p == 0) & (m == 0)] = 1  # flat
        geom[(p == 0) & (m >= 5)] = 2  # peak
        geom[(p == 0) & (0 < m) & (m < 5)] = 3  # ridge
        geom[(m == 0) & (0 < p) & (p < 5)] = 9  # valley
        geom[(m == 0) & (p >= 5)] = 10  # pit
        geom[(m > p) & (p > 0) & ((m - p) >= 3)] = 4  # shoulder
        geom[(m > p) & (p > 0) & ((m - p) < 3)] = 5  # spur
        geom[(p > m) & (m > 0) & ((p - m) >= 3)] = 8  # footslope
        geom[(p > m) & (m > 0) & ((p - m) < 3)] = 7  # hollow

        return xr.DataArray(
            geom, coords=self.data.coords, dims=self.data.dims, name="geomorphons"
        )

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
        return Profile(
            self.data,
            start_lon,
            start_lat,
            end_lon,
            end_lat,
            num_points,
            point_spacing,
            name,
        )

    # Plotting methods

    def plot_bathy(
        self, contours: int | list[float] | None = None, cmap=None, **kwargs
    ) -> None:
        """
        Plot bathymetry elevation.

        Parameters
        ----------
        contours : int or list[float], optional
            If int, number of contour levels to plot
            If list, specific contour levels (in meters)
            If None, no contours are plotted
        cmap : str or Colormap, optional
            Colormap to use. Defaults to cmocean 'deep_r'
            (light=shallow, dark=deep)
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

    def plot_hillshade(
        self,
        azimuth: float = 315,
        altitude: float = 45,
        contours: int | list[float] | None = None,
        **kwargs,
    ) -> None:
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
        ax.imshow(
            shaded, cmap="gray", origin="lower", extent=extent, aspect="auto", **kwargs
        )

        if contours is not None:
            self._add_contours(ax, contours)

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        plt.show()

    def plot_slope(
        self,
        contours: int | list[float] | None = None,
        vmax: float | None = None,
        **kwargs,
    ) -> None:
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

    def plot_curvature(
        self, contours: int | list[float] | None = None, **kwargs
    ) -> None:
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

    def plot_bpi(
        self,
        radius_km: float = 1.0,
        contours: int | list[float] | None = None,
        **kwargs,
    ) -> None:
        """
        Plot Bathymetric Position Index (BPI).

        Parameters
        ----------
        radius_km : float, default 1.0
            Radius of the neighbourhood in kilometres (square window)
        contours : int or list[float], optional
            If int, number of contour levels to plot
            If list, specific contour levels (in metres)
            If None, no contours are plotted
        **kwargs
            Additional arguments passed to imshow
        """
        bpi_data = self.bpi(radius_km)
        extent = get_extent(self.data)

        fig, ax = plt.subplots(figsize=(10, 8))

        vmax = np.nanmax(np.abs(bpi_data.values))
        im = ax.imshow(
            bpi_data.values,
            cmap=cmo.balance,
            origin="lower",
            extent=extent,
            aspect="auto",
            vmin=-vmax,
            vmax=vmax,
            **kwargs,
        )
        plt.colorbar(im, ax=ax, label=f"BPI (r={radius_km} km)")

        if contours is not None:
            self._add_contours(ax, contours)

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        plt.show()

    def plot_rugosity(
        self,
        radius_km: float = 1.0,
        contours: int | list[float] | None = None,
        vmax: float | None = None,
        **kwargs,
    ) -> None:
        """
        Plot Vector Ruggedness Measure (VRM).

        Parameters
        ----------
        radius_km : float, default 1.0
            Radius of the neighbourhood in kilometres
        contours : int or list[float], optional
            If int, number of contour levels to plot
            If list, specific contour levels (in metres)
            If None, no contours are plotted
        vmax : float, optional
            Maximum VRM value for colour scale.
            Defaults to the 99th percentile to avoid outliers dominating.
        **kwargs
            Additional arguments passed to imshow
        """
        rug_data = self.rugosity(radius_km)
        extent = get_extent(self.data)

        if vmax is None:
            vmax = float(np.nanpercentile(rug_data.values, 99))

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            rug_data.values,
            cmap=cmo.amp,
            origin="lower",
            extent=extent,
            aspect="auto",
            vmin=0,
            vmax=vmax,
            **kwargs,
        )
        plt.colorbar(im, ax=ax, label=f"Rugosity VRM (r={radius_km} km)")

        if contours is not None:
            self._add_contours(ax, contours)

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        plt.show()

    def plot_aspect(
        self,
        contours: int | list[float] | None = None,
        **kwargs,
    ) -> None:
        """
        Plot seafloor aspect.

        Uses a circular colormap so that north (0°) and north (360°) share
        the same colour.

        Parameters
        ----------
        contours : int or list[float], optional
            If int, number of contour levels to plot
            If list, specific contour levels (in metres)
            If None, no contours are plotted
        **kwargs
            Additional arguments passed to imshow
        """
        asp_data = self.aspect()
        extent = get_extent(self.data)

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
            self._add_contours(ax, contours)

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        plt.show()

    def plot_geomorphons(
        self,
        lookup_km: float = 2.0,
        flatness_threshold: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Plot seafloor morphology using geomorphons.

        Parameters
        ----------
        lookup_km : float, default 2.0
            Lookup distance in kilometres (passed to geomorphons()).
        flatness_threshold : float, default 1.0
            Flatness angle threshold in degrees (passed to geomorphons()).
        **kwargs
            Additional arguments passed to imshow.

        Examples
        --------
        >>> bath.plot_geomorphons(lookup_km=2.0)
        """
        geom_data = self.geomorphons(lookup_km, flatness_threshold)
        extent = get_extent(self.data)

        labels = [
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
        # Warm (elevated) → grey (neutral) → cool (depressed) scheme
        colors = [
            "#d9d9d9",  # flat      — medium grey
            "#7a0000",  # peak      — dark red-brown
            "#c0392b",  # ridge     — muted red
            "#e07b6b",  # shoulder  — muted salmon
            "#e8a45a",  # spur      — muted orange
            "#909090",  # slope     — mid-grey
            "#5dade2",  # hollow    — muted sky blue
            "#2980b9",  # footslope — medium blue
            "#1a5276",  # valley    — dark teal-blue
            "#0b2545",  # pit       — near-black navy
        ]

        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(0.5, 11.5), len(colors))

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

        cbar = plt.colorbar(im, ax=ax, label="Geomorphon")
        cbar.set_ticks(range(1, 11))
        cbar.set_ticklabels(labels)

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
        im = ax.imshow(
            self.data.values,
            cmap=colors,
            norm=norm,
            origin="lower",
            extent=extent,
            aspect="auto",
            **kwargs,
        )

        if contours is not None:
            self._add_contours(ax, contours)

        cbar = plt.colorbar(im, ax=ax, label="Depth zone")

        # Set ticks at the center of each color band with depth range labels
        tick_positions = [
            (boundaries[i] + boundaries[i + 1]) / 2 for i in range(n_zones)
        ]
        tick_labels = [
            f"{reversed_labels[i]}\n"
            f"({int(boundaries[i + 1])} to {int(boundaries[i])} m)"
            for i in range(n_zones)
        ]
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

        # Set aspect ratio accounting for longitude compression at higher latitudes
        lat_centre = float(self.data.lat.mean())
        lon_scale = np.cos(np.radians(lat_centre))
        ax.set_box_aspect(
            [
                np.ptp(lon) * lon_scale,
                np.ptp(lat),
                np.ptp(z) * vertical_exaggeration / 1000,
            ]
        )

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
        return (
            f"Bathymetry(shape={self.shape}, "
            f"lon=[{lon_min:.2f}, {lon_max:.2f}], "
            f"lat=[{lat_min:.2f}, {lat_max:.2f}])"
        )
