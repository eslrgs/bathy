# Loading Data

bathy supports multiple data sources and formats for bathymetric data.

## GEBCO OPeNDAP

You can download data directly from the GEBCO global bathymetry dataset via OPeNDAP:

```python
import bathy

data = bathy.load_gebco_opendap(
    lon_range=(-12, -5),
    lat_range=(46, 50),
    year=2025,  # GEBCO version year
    save_path="data/region.nc",  # Optional: cache locally
)
```

### Caching behaviour

If `save_path` is provided and the file already exists, bathy will load from the file instead of downloading. This avoids redundant downloads.

If `save_path` is omitted, data is downloaded to a temporary file that is automatically deleted after loading. For large regions, provide `save_path` to keep the file.

### Download size

Download size scales with the requested area. The full GEBCO global grid is ~8 GB. bathy estimates the download size and logs it during download. For regions estimated at over 500 MB, `save_path` is required — this prevents accidentally filling the system temp directory with large files that are immediately deleted.

### Valid years

GEBCO datasets are available for the following years: 2019-2025. An invalid year will raise a `ValueError`.

### Preset regions

Common oceanographic regions are available as presets:

```python
import bathy

# List all available regions
regions = bathy.list_regions()
print(regions[:10])
# ['arabian_sea', 'baltic_sea', 'bay_of_bengal', 'black_sea', ...]

# Use a preset
data = bathy.load_gebco_opendap(region="mediterranean")
```

## EMODnet Bathymetry (European seas)

EMODnet provides high-resolution (~115 m) gridded bathymetry for European maritime areas via a Web Coverage Service (WCS):

```python
import bathy

data = bathy.load_emodnet_wcs(
    lon_range=(-10, -5),
    lat_range=(50, 55),
    save_path="data/emodnet_region.tif",  # Optional: cache locally
)
```

This returns a GeoTIFF-backed `xr.DataArray` with `lon`/`lat` coordinates. The same preset regions and caching behaviour as GEBCO apply:

```python
data = bathy.load_emodnet_wcs(region="north_sea")
```

!!! note
    EMODnet coverage is limited to European seas. Requests outside this area will return an error.

The same download size estimation and `save_path` requirement for large regions applies to EMODnet downloads.

## Local NetCDF files

Load from local NetCDF files (e.g., downloaded GEBCO data):

```python
data = bathy.load_bathymetry(
    "path/to/gebco_2025.nc",
    lon_range=(-10, -5),
    lat_range=(50, 55),
    var_name="elevation",  # Variable name in file
    lon_name="lon",        # Longitude coordinate name
    lat_name="lat",        # Latitude coordinate name
)
```

## GeoTIFF files

Load from GeoTIFF rasters:

```python
data = bathy.load_bathymetry("path/to/bathymetry.tif")
```

GeoTIFF files are loaded using rioxarray. The CRS is read automatically:

- **Geographic CRS** (e.g. EPSG:4326): coordinates are renamed to `lon`/`lat`
- **Projected CRS** (e.g. UTM): coordinates are kept as `x`/`y` in metres

All analysis and plotting functions adapt to the CRS automatically. See [Projected Coordinate Systems](projections.md) for details.

## Clipping to a region

`data` is an `xr.DataArray`, so xarray's standard selection works directly:

```python
subset = data.sel(lon=slice(-10, -7), lat=slice(47, 49))
```

## Exporting data

### To GeoTIFF

```python
bathy.to_geotiff(data, "output.tif", crs="EPSG:4326")
```

### To NetCDF

```python
data.to_netcdf("output.nc")
```

## Inspecting data

`data` is a standard `xr.DataArray` — use xarray directly:

```python
print(data.shape)   # Grid dimensions
print(data.dims)    # ('lat', 'lon') or ('y', 'x')
print(data.coords)  # All coordinates
data                 # Rich repr in Jupyter
```
