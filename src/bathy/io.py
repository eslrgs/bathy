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

_LARGE_DOWNLOAD_MB = 500


def _estimate_download_size_mb(
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    resolution_deg: float,
    bytes_per_pixel: int = 4,
) -> float:
    """Estimate download size in MB from bounding box and grid resolution."""
    lon_span = abs(max(lon_range) - min(lon_range))
    lat_span = abs(max(lat_range) - min(lat_range))
    n_pixels = (lon_span / resolution_deg) * (lat_span / resolution_deg)
    return (n_pixels * bytes_per_pixel) / (1024 * 1024)


def _warn_if_large(
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    resolution_deg: float,
    save_path: str | None,
) -> float:
    """Estimate download size, warn or error if large. Returns estimated MB."""
    estimated_mb = _estimate_download_size_mb(lon_range, lat_range, resolution_deg)
    if estimated_mb > _LARGE_DOWNLOAD_MB:
        if save_path is None:
            raise ValueError(
                f"Estimated download size is ~{estimated_mb:.0f} MB. "
                f"For large regions, provide 'save_path' to avoid downloading "
                f"to a temporary file that is deleted after loading."
            )
        logger.warning(
            f"Large download: estimated ~{estimated_mb:.0f} MB. This may take a while."
        )
    return estimated_mb


def _download_with_progress(
    url: str,
    save_path: str | None,
    *,
    suffix: str = ".nc",
    desc: str = "Downloading",
    timeout: int = 600,
) -> str:
    """Download URL to a file with a tqdm progress bar."""
    if save_path is None:
        fd, filepath = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
    else:
        filepath = save_path

    try:
        response = urlopen(url, timeout=timeout)  # noqa: S310
        total = int(response.headers.get("Content-Length", 0))

        with (
            open(filepath, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar,
        ):
            while chunk := response.read(8192):
                f.write(chunk)
                pbar.update(len(chunk))
    except Exception:
        if save_path is None:
            Path(filepath).unlink(missing_ok=True)
        raise

    logger.info(f"Saved to {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Dataset-specific download helpers
# ---------------------------------------------------------------------------

_GEBCO_VALID_YEARS = {2019, 2020, 2021, 2022, 2023, 2024, 2025}


def _download_gebco(
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    year: int,
    save_path: str | None,
) -> str:
    """Download GEBCO data from THREDDS server."""
    if year not in _GEBCO_VALID_YEARS:
        raise ValueError(
            f"Invalid GEBCO year: {year}. Valid years: {sorted(_GEBCO_VALID_YEARS)}"
        )

    estimated_mb = _warn_if_large(lon_range, lat_range, 1 / 240, save_path)

    params = {
        "var": "elevation",
        "north": max(lat_range),
        "south": min(lat_range),
        "west": min(lon_range),
        "east": max(lon_range),
    }
    base_url = (
        f"https://dap.ceda.ac.uk/thredds/ncss/bodc/gebco/global/"
        f"gebco_{year}/ice_surface_elevation/netcdf/GEBCO_{year}.nc"
    )
    url = f"{base_url}?{urlencode(params)}"

    logger.info(
        f"Downloading GEBCO {year} from CEDA (estimated ~{estimated_mb:.0f} MB)..."
    )
    return _download_with_progress(url, save_path, desc="Downloading GEBCO")


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

    estimated_mb = _warn_if_large(lon_range, lat_range, 1 / 480, save_path)

    if save_path is None:
        fd, filepath = tempfile.mkstemp(suffix=".tif")
        os.close(fd)
    else:
        filepath = save_path

    logger.info(f"Downloading EMODnet bathymetry (estimated ~{estimated_mb:.0f} MB)...")

    try:
        wcs = WebCoverageService(_EMODNET_WCS_URL, version="1.0.0", timeout=600)

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
        if save_path is None:
            Path(filepath).unlink(missing_ok=True)
        raise

    logger.info(f"Saved to {filepath}")
    return filepath


_ETOPO_OPENDAP_BASE = "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022"


# NOAA CRM volume bounding boxes: (lon_min, lon_max, lat_min, lat_max)
_CRM_VOLUMES: dict[int, tuple[float, float, float, float]] = {
    1: (-80, -64, 40, 48),
    2: (-85, -68, 31, 40),
    3: (-87, -78, 24, 35),
    4: (-94, -87, 24, 36),
    5: (-108, -94, 24, 38),
    6: (-126, -114, 32, 37),
    7: (-128, -117, 37, 44),
    8: (-128, -116, 44, 49),
    9: (-68, -64, 16, 20),  # Puerto Rico / USVI
    10: (-162, -152, 18, 24),  # Hawaii
}


def _find_crm_volume(
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
) -> int:
    """Find the CRM volume that best overlaps the requested region."""
    lon_min, lon_max = min(lon_range), max(lon_range)
    lat_min, lat_max = min(lat_range), max(lat_range)

    best_vol = None
    best_overlap = 0.0

    for vol, (vlon_min, vlon_max, vlat_min, vlat_max) in _CRM_VOLUMES.items():
        # Intersection area
        dx = max(0, min(lon_max, vlon_max) - max(lon_min, vlon_min))
        dy = max(0, min(lat_max, vlat_max) - max(lat_min, vlat_min))
        overlap = dx * dy
        if overlap > best_overlap:
            best_overlap = overlap
            best_vol = vol

    if best_vol is None or best_overlap == 0:
        raise ValueError(
            "Requested region does not overlap any NOAA CRM volume. "
            "CRM covers US coastal waters only."
        )
    return best_vol


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
        GEBCO dataset year. Valid years: 2019-2025.
    save_path : str, optional
        If provided, save the downloaded file to this path for reuse.
        If omitted, the data is downloaded to a temporary file that is
        automatically deleted after loading.

    Returns
    -------
    xr.DataArray
        Elevation data

    Notes
    -----
    Download size scales with the requested area. The full global grid is
    ~8 GB. For regions larger than ~500 MB, ``save_path`` is required to
    avoid downloading to a temporary file. Large downloads will log a
    warning with the estimated size.

    References
    ----------
    GEBCO Compilation Group (2025) GEBCO 2025 Grid
    (doi:10.5285/37c52e96-24ea-67ce-e063-7086abc05f29)

    Examples
    --------
    >>> data = load_gebco_opendap(lon_range=(-10, -5), lat_range=(50, 55))
    >>> data = load_gebco_opendap(region='mediterranean')
    >>> # Large region — use save_path to keep the file
    >>> data = load_gebco_opendap(region='pacific', save_path='pacific.nc')
    """
    lon_range, lat_range = _resolve_region(
        lon_range, lat_range, region, require_bounds=True
    )
    assert lon_range is not None and lat_range is not None  # noqa: S101

    if save_path and os.path.exists(save_path):
        logger.info(f"Using existing file: {save_path}")
        filepath = save_path
    else:
        filepath = _download_gebco(lon_range, lat_range, year, save_path)

    try:
        return load_bathymetry(filepath)
    finally:
        if save_path is None:
            Path(filepath).unlink(missing_ok=True)


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
        If provided, save the downloaded GeoTIFF to this path for reuse.
        If the file already exists, it is loaded without downloading.
        If omitted, the data is downloaded to a temporary file that is
        automatically deleted after loading.

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
    assert lon_range is not None and lat_range is not None  # noqa: S101

    if save_path and os.path.exists(save_path):
        logger.info(f"Using existing file: {save_path}")
        filepath = save_path
    else:
        filepath = _download_emodnet(lon_range, lat_range, save_path)

    try:
        return load_bathymetry(filepath)
    finally:
        if save_path is None:
            Path(filepath).unlink(missing_ok=True)


def load_etopo(
    lon_range: tuple[float, float] | None = None,
    lat_range: tuple[float, float] | None = None,
    region: str | None = None,
    resolution: str = "60s",
    save_path: str | None = None,
) -> xr.DataArray:
    """
    Download NOAA ETOPO 2022 global relief data.

    ETOPO provides integrated topography and bathymetry from NOAA NCEI,
    widely used by US-based and global researchers.

    Parameters
    ----------
    lon_range : tuple[float, float], optional
        Longitude bounds (min, max). Cannot be used with 'region'.
    lat_range : tuple[float, float], optional
        Latitude bounds (min, max). Cannot be used with 'region'.
    region : str, optional
        Preset region name. See `bathy.list_regions()`.
        Cannot be used with 'lon_range' or 'lat_range'.
    resolution : str, default '60s'
        Grid resolution: '60s' (1 arc-minute), '30s', or '15s'.
    save_path : str, optional
        If provided, save the downloaded file to this path for reuse.
        If the file already exists, it is loaded without downloading.

    Returns
    -------
    xr.DataArray
        Elevation data with 'lon' and 'lat' coordinates

    References
    ----------
    NOAA National Centers for Environmental Information (2022).
    ETOPO 2022 15 Arc-Second Global Relief Model.
    https://doi.org/10.25921/fd45-gt74

    Examples
    --------
    >>> data = load_etopo(lon_range=(-10, -5), lat_range=(50, 55))
    >>> data = load_etopo(region='mediterranean', resolution='30s')
    """
    valid_resolutions = {"60s", "30s", "15s"}
    if resolution not in valid_resolutions:
        raise ValueError(
            f"Invalid resolution: {resolution}. Valid: {sorted(valid_resolutions)}"
        )

    lon_range, lat_range = _resolve_region(
        lon_range, lat_range, region, require_bounds=True
    )
    assert lon_range is not None and lat_range is not None  # noqa: S101

    if save_path and os.path.exists(save_path):
        logger.info(f"Using existing file: {save_path}")
        return _load_netcdf(save_path, lon_range, lat_range, "z", "lon", "lat")

    opendap_url = (
        f"{_ETOPO_OPENDAP_BASE}/{resolution}/"
        f"{resolution}_surface_elev_netcdf/"
        f"ETOPO_2022_v1_{resolution}_N90W180_surface.nc"
    )

    logger.info(f"Accessing ETOPO 2022 ({resolution}) via OPeNDAP...")

    try:
        with xr.open_dataset(opendap_url, engine="pydap") as ds:
            data = (
                ds["z"]
                .sel(lon=slice(*sorted(lon_range)), lat=slice(*sorted(lat_range)))
                .load()
            )
    except OSError as e:
        raise ConnectionError(
            f"Failed to access ETOPO OPeNDAP server. "
            f"Check your internet connection or try again later. "
            f"URL: {opendap_url}"
        ) from e

    if data.size == 0:
        raise ValueError(
            f"Data selection resulted in empty array. "
            f"Requested: lon={lon_range}, lat={lat_range}."
        )

    if save_path:
        data.to_netcdf(save_path)
        logger.info(f"Saved to {save_path}")

    return data


def load_noaa_crm(
    lon_range: tuple[float, float] | None = None,
    lat_range: tuple[float, float] | None = None,
    region: str | None = None,
    save_path: str | None = None,
) -> xr.DataArray:
    """
    Load NOAA Coastal Relief Model (~3 arc-second / ~90 m) via OPeNDAP.

    High-resolution bathymetry/topography for US coastal waters. The
    correct CRM volume is selected automatically based on the requested
    region.

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
        If provided, save the subsetted data as NetCDF for reuse.
        If the file already exists, it is loaded without fetching.

    Returns
    -------
    xr.DataArray
        Elevation data with 'lon' and 'lat' coordinates

    Notes
    -----
    Coverage is limited to US coastal waters (10 regional volumes).
    Requesting a region outside US waters will raise a ``ValueError``.

    References
    ----------
    NOAA National Centers for Environmental Information.
    U.S. Coastal Relief Model.
    https://www.ngdc.noaa.gov/mgg/coastal/crm.html

    Examples
    --------
    >>> data = load_noaa_crm(lon_range=(-72, -70), lat_range=(41, 43))
    >>> data = load_noaa_crm(region='us_east_coast', save_path='crm.nc')
    """
    lon_range, lat_range = _resolve_region(
        lon_range, lat_range, region, require_bounds=True
    )
    assert lon_range is not None and lat_range is not None  # noqa: S101

    if save_path and os.path.exists(save_path):
        logger.info(f"Using existing file: {save_path}")
        return _load_netcdf(save_path, lon_range, lat_range, "z", "lon", "lat")

    vol = _find_crm_volume(lon_range, lat_range)
    opendap_url = f"https://www.ngdc.noaa.gov/thredds/dodsC/crm/crm_vol{vol}.nc"

    logger.info(f"Accessing NOAA CRM volume {vol} via OPeNDAP...")

    try:
        with xr.open_dataset(opendap_url, engine="pydap") as ds:
            # CRM uses x/y coordinate names (geographic degrees despite the names)
            data = (
                ds["z"]
                .sel(x=slice(*sorted(lon_range)), y=slice(*sorted(lat_range)))
                .load()
            )
    except OSError as e:
        raise ConnectionError(
            f"Failed to access NOAA CRM OPeNDAP server. "
            f"Check your internet connection or try again later. "
            f"URL: {opendap_url}"
        ) from e

    if data.size == 0:
        raise ValueError(
            f"Data selection resulted in empty array. "
            f"Requested: lon={lon_range}, lat={lat_range}. "
            f"CRM volume {vol} may not fully cover this region."
        )

    # Rename x/y → lon/lat to match bathy conventions
    data = data.rename({"x": "lon", "y": "lat"})

    if save_path:
        data.to_netcdf(save_path)
        logger.info(f"Saved to {save_path}")

    return data


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
