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


def _resolve_region(
    lon_range: tuple[float, float] | None,
    lat_range: tuple[float, float] | None,
    region: str | None,
    *,
    require_bounds: bool = False,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Resolve region preset to lon/lat ranges, with validation."""
    if region is not None:
        if lon_range is not None or lat_range is not None:
            raise ValueError(
                "Cannot specify both 'region' and 'lon_range'/'lat_range'."
            )
        lon_min, lon_max, lat_min, lat_max = _get_region(region)
        lon_range = (lon_min, lon_max)
        lat_range = (lat_min, lat_max)

    if require_bounds and (lon_range is None or lat_range is None):
        raise ValueError(
            "Must specify either 'region' or both 'lon_range' and 'lat_range'"
        )

    return lon_range, lat_range


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
        if os.path.exists(filepath) and save_path is None:
            os.unlink(filepath)
        raise

    logger.info(f"Saved to {filepath}")

    return filepath


_EMODNET_WCS_URL = "https://ows.emodnet-bathymetry.eu/wcs"
_EMODNET_COVERAGE = "emodnet:mean"


def _download_emodnet(
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    save_path: str | None,
) -> str:
    """Download EMODnet bathymetry data via OWSLib WCS."""
    from owslib.wcs import WebCoverageService  # noqa: PLC0415

    lon_min, lon_max = min(lon_range), max(lon_range)
    lat_min, lat_max = min(lat_range), max(lat_range)

    if save_path is None:
        fd, filepath = tempfile.mkstemp(suffix=".tif")
        os.close(fd)
    else:
        filepath = save_path

    logger.info("Downloading EMODnet bathymetry data...")

    try:
        wcs = WebCoverageService(_EMODNET_WCS_URL, version="1.0.0", timeout=120)

        response = wcs.getCoverage(
            identifier=_EMODNET_COVERAGE,
            bbox=(lon_min, lat_min, lon_max, lat_max),
            crs="EPSG:4326",
            format="image/tiff",
            resx=0.00208333,
            resy=0.00208333,
        )

        data = response.read()

        if len(data) == 0:
            raise ValueError(
                "EMODnet WCS returned empty response. "
                "Check that your region overlaps European seas."
            )

        # WCS may return XML error instead of TIFF
        if data[:5] != b"II\x2a\x00\x08" and data[:4] != b"MM\x00\x2a":
            body = data.decode("utf-8", errors="replace")[:500]
            raise ValueError(
                f"EMODnet WCS returned an error. "
                f"Check that your region overlaps European seas. "
                f"Response: {body}"
            )

        with open(filepath, "wb") as f:
            f.write(data)

    except Exception:
        if os.path.exists(filepath) and save_path is None:
            os.unlink(filepath)
        raise

    logger.info(f"Saved to {filepath}")

    return filepath


def _load_geotiff(
    filepath: str | Path,
    lon_range: tuple[float, float] | None,
    lat_range: tuple[float, float] | None,
) -> xr.DataArray:
    """Load GeoTIFF file with rioxarray.

    For multi-band files, only band 1 is selected.
    """
    import rioxarray  # noqa: PLC0415

    da: xr.DataArray = rioxarray.open_rasterio(filepath, masked=True)  # ty: ignore[invalid-assignment]

    if "band" in da.dims:
        da = da.sel(band=1)

    # Only rename x/y → lon/lat for geographic CRS; keep x/y for projected
    if "x" in da.dims and "y" in da.dims:
        crs = da.rio.crs
        if crs is None or crs.is_geographic:
            da = da.rename({"x": "lon", "y": "lat"})

    from bathy.utils import get_dim_names  # noqa: PLC0415

    x_dim, y_dim = get_dim_names(da)

    if lon_range is not None:
        lo, hi = min(lon_range), max(lon_range)
        da = da.where((da[x_dim] >= lo) & (da[x_dim] <= hi), drop=True)
    if lat_range is not None:
        lo, hi = min(lat_range), max(lat_range)
        da = da.where((da[y_dim] >= lo) & (da[y_dim] <= hi), drop=True)

    # Ensure ascending coordinate order for consistent slicing
    for dim in (x_dim, y_dim):
        if da[dim].size > 1 and float(da[dim][0]) > float(da[dim][-1]):
            da = da.sortby(dim)

    return da


def _load_netcdf(
    filepath: str | Path,
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
    lon_range, lat_range = _resolve_region(lon_range, lat_range, region)

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

    References
    ----------
    GEBCO Compilation Group (2025) GEBCO 2025 Grid
    (doi:10.5285/37c52e96-24ea-67ce-e063-7086abc05f29)

    Examples
    --------
    >>> data = load_gebco_opendap(lon_range=(-10, -5), lat_range=(50, 55))
    >>> data = load_gebco_opendap(region='mediterranean')
    """
    lon_range, lat_range = _resolve_region(
        lon_range, lat_range, region, require_bounds=True
    )

    assert lon_range is not None and lat_range is not None

    if save_path and os.path.exists(save_path):
        logger.info(f"Using existing file: {save_path}")
        filepath = save_path
    else:
        filepath = _download_gebco(lon_range, lat_range, year, save_path)

    return load_bathymetry(filepath)


def load_emodnet_wcs(
    lon_range: tuple[float, float] | None = None,
    lat_range: tuple[float, float] | None = None,
    region: str | None = None,
    save_path: str | None = None,
) -> xr.DataArray:
    """
    Download bathymetry from the EMODnet Web Coverage Service.

    EMODnet provides high-resolution (~115 m) gridded bathymetry for
    European seas. Coverage is limited to European maritime areas.

    Parameters
    ----------
    lon_range : tuple[float, float], optional
        Longitude bounds (min, max). Cannot be used with 'region'.
    lat_range : tuple[float, float], optional
        Latitude bounds (min, max). Cannot be used with 'region'.
    region : str, optional
        Preset region name. See `bathy.list_regions()`.
        Cannot be used with 'lon_range' or 'lat_range'.
    save_path : str, optional
        If provided, save the downloaded GeoTIFF to this path.
        If the file already exists, it is loaded without downloading.

    Returns
    -------
    xr.DataArray
        Elevation data with 'lon' and 'lat' coordinates

    References
    ----------
    EMODnet Bathymetry Consortium (2024). EMODnet Digital Bathymetry (DTM).
    https://emodnet.ec.europa.eu/en/bathymetry

    Examples
    --------
    >>> data = load_emodnet_wcs(lon_range=(-10, -5), lat_range=(50, 55))
    >>> data = load_emodnet_wcs(region='north_sea')
    """
    lon_range, lat_range = _resolve_region(
        lon_range, lat_range, region, require_bounds=True
    )

    assert lon_range is not None and lat_range is not None

    if save_path and os.path.exists(save_path):
        logger.info(f"Using existing file: {save_path}")
        filepath = save_path
    else:
        filepath = _download_emodnet(lon_range, lat_range, save_path)

    return load_bathymetry(filepath)


def to_geotiff(
    data: xr.DataArray,
    filepath: str | Path,
    crs: str | None = None,
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
    crs : str, optional
        Coordinate reference system.  Only used when the data has no CRS
        attached; defaults to ``"EPSG:4326"`` for geographic data.
    **kwargs
        Additional arguments passed to rioxarray.to_raster()

    Examples
    --------
    >>> to_geotiff(data, 'output.tif')
    """
    parent = Path(filepath).parent
    if not parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {parent}")

    if data.rio.crs is None:
        data = data.rio.write_crs(crs or "EPSG:4326")
    data.rio.to_raster(filepath, **kwargs)
