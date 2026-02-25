"""Bathymetry data loading and exporting."""

import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import xarray as xr
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Preset regions dictionary: {name: (lon_min, lon_max, lat_min, lat_max)}
REGIONS = {
    # Atlantic Ocean
    "north_atlantic": (-80, 0, 40, 70),
    "mid_atlantic_ridge": (-45, -15, -30, 30),
    "gulf_of_mexico": (-98, -80, 18, 31),
    "caribbean": (-90, -60, 10, 25),
    # Pacific Ocean
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
    "greenland": (-55, -20, 60, 83),
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
    ['arabian_sea', 'baltic_sea', 'bay_of_bengal', 'black_sea', 'caribbean']
    """
    return sorted(REGIONS.keys())


def _get_region(name: str) -> tuple[float, float, float, float]:
    """Get coordinates for a preset region."""
    if name not in REGIONS:
        available = ", ".join(list_regions()[:5])
        raise ValueError(
            f"Region '{name}' not found. "
            f"Available: {available}, ... "
            f"(see bathy.list_regions())"
        )
    return REGIONS[name]


# ============================================================================
# Internal helpers
# ============================================================================


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

    if save_path is None:
        fd, filepath = tempfile.mkstemp(suffix=".nc")
        os.close(fd)
    else:
        filepath = save_path

    logger.info(f"Downloading GEBCO {year} data from CEDA...")

    try:
        response = urlopen(ncss_url, timeout=120)  # noqa: S310
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
    except Exception:
        if os.path.exists(filepath):
            os.unlink(filepath)
        raise

    logger.info(f"Saved to {filepath}")

    return filepath


def _load_geotiff(
    filepath: str | Path,
    lon_range: tuple[float, float] | None,
    lat_range: tuple[float, float] | None,
) -> xr.DataArray:
    """Load GeoTIFF file with rioxarray."""
    import rioxarray  # noqa: PLC0415

    da = rioxarray.open_rasterio(filepath, masked=True)

    if "band" in da.dims:
        da = da.sel(band=1)

    if "x" in da.dims and "y" in da.dims:
        da = da.rename({"x": "lon", "y": "lat"})

    if lon_range is not None:
        lon_min, lon_max = min(lon_range), max(lon_range)
        da = da.where((da.lon >= lon_min) & (da.lon <= lon_max), drop=True)
    if lat_range is not None:
        lat_min, lat_max = min(lat_range), max(lat_range)
        da = da.where((da.lat >= lat_min) & (da.lat <= lat_max), drop=True)

    return da


def _load_netcdf(
    filepath: str,
    lon_range: tuple[float, float] | None,
    lat_range: tuple[float, float] | None,
    var_name: str,
    lon_name: str,
    lat_name: str,
) -> xr.DataArray:
    """Load NetCDF file with xarray."""
    with xr.open_dataset(filepath) as ds:
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

        original_lon_bounds = (float(ds[lon_name].min()), float(ds[lon_name].max()))
        original_lat_bounds = (float(ds[lat_name].min()), float(ds[lat_name].max()))

        if lon_range:
            ds = ds.sel({lon_name: slice(*lon_range)})
        if lat_range:
            ds = ds.sel({lat_name: slice(*sorted(lat_range))})

        data = ds[var_name].load()

    rename_dict = {}
    if lon_name != "lon":
        rename_dict[lon_name] = "lon"
    if lat_name != "lat":
        rename_dict[lat_name] = "lat"
    if rename_dict:
        data = data.rename(rename_dict)

    if data.size == 0:
        raise ValueError(
            f"Data selection resulted in empty array. "
            f"Requested: lon={lon_range}, lat={lat_range}. "
            f"Available: lon={original_lon_bounds}, lat={original_lat_bounds}"
        )

    return data


# ============================================================================
# Public functions
# ============================================================================


def load_bathymetry(
    filepath: str | Path,
    lon_range: tuple[float, float] | None = None,
    lat_range: tuple[float, float] | None = None,
    region: str | None = None,
    var_name: str = "elevation",
    lon_name: str = "lon",
    lat_name: str = "lat",
) -> xr.DataArray:
    """
    Load bathymetry data from a NetCDF or GeoTIFF file.

    Parameters
    ----------
    filepath : str
        Path to the file
    lon_range : tuple[float, float], optional
        Longitude bounds (min, max). Cannot be used with 'region'.
    lat_range : tuple[float, float], optional
        Latitude bounds (min, max). Cannot be used with 'region'.
    region : str, optional
        Preset region name. See `bathy.list_regions()`.
        Cannot be used with 'lon_range' or 'lat_range'.
    var_name : str, default 'elevation'
        Variable name
    lon_name : str, default 'lon'
        Longitude coordinate name
    lat_name : str, default 'lat'
        Latitude coordinate name

    Returns
    -------
    xr.DataArray
        Elevation data with 'lon' and 'lat' coordinates

    Examples
    --------
    >>> data = load_bathymetry('gebco.nc', lon_range=(-10, -5), lat_range=(50, 55))
    >>> data = load_bathymetry('gebco.nc', region='mediterranean')
    """
    if region is not None:
        if lon_range is not None or lat_range is not None:
            raise ValueError(
                "Cannot specify both 'region' and 'lon_range'/'lat_range'."
            )
        lon_min, lon_max, lat_min, lat_max = _get_region(region)
        lon_range = (lon_min, lon_max)
        lat_range = (lat_min, lat_max)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".tif", ".tiff"]:
        return _load_geotiff(filepath, lon_range, lat_range)
    return _load_netcdf(filepath, lon_range, lat_range, var_name, lon_name, lat_name)


def load_gebco_opendap(
    lon_range: tuple[float, float] | None = None,
    lat_range: tuple[float, float] | None = None,
    region: str | None = None,
    year: int = 2025,
    save_path: str | None = None,
) -> xr.DataArray:
    """
    Download GEBCO data from OPeNDAP server for a specific region.

    Parameters
    ----------
    lon_range : tuple[float, float], optional
        Longitude bounds (min, max), range: -180 to 180.
        Cannot be used with 'region'.
    lat_range : tuple[float, float], optional
        Latitude bounds (min, max), range: -90 to 90.
        Cannot be used with 'region'.
    region : str, optional
        Preset region name. See `bathy.list_regions()`.
        Cannot be used with 'lon_range' or 'lat_range'.
    year : int, default 2025
        GEBCO dataset year
    save_path : str, optional
        If provided, save the downloaded data to this path

    Returns
    -------
    xr.DataArray
        Elevation data

    Examples
    --------
    >>> data = load_gebco_opendap(lon_range=(-10, -5), lat_range=(50, 55))
    >>> data = load_gebco_opendap(region='mediterranean')
    """
    if region is not None:
        if lon_range is not None or lat_range is not None:
            raise ValueError(
                "Cannot specify both 'region' and 'lon_range'/'lat_range'."
            )
        lon_min, lon_max, lat_min, lat_max = _get_region(region)
        lon_range = (lon_min, lon_max)
        lat_range = (lat_min, lat_max)

    if lon_range is None or lat_range is None:
        raise ValueError(
            "Must specify either 'region' or both 'lon_range' and 'lat_range'"
        )

    if save_path and os.path.exists(save_path):
        logger.info(f"Using existing file: {save_path}")
        filepath = save_path
    else:
        filepath = _download_gebco(lon_range, lat_range, year, save_path)

    return load_bathymetry(filepath)


def to_geotiff(
    data: xr.DataArray,
    filepath: str | Path,
    crs: str = "EPSG:4326",
    **kwargs,
) -> None:
    """
    Save data to a GeoTIFF file.

    Parameters
    ----------
    data : xr.DataArray
        Elevation data
    filepath : str or Path
        Output GeoTIFF file path
    crs : str, default 'EPSG:4326'
        Coordinate reference system
    **kwargs
        Additional arguments passed to rioxarray.to_raster()

    Examples
    --------
    >>> to_geotiff(data, 'output.tif')
    """

    if data.rio.crs is None:
        data = data.rio.write_crs(crs)
    data.rio.to_raster(filepath, **kwargs)
