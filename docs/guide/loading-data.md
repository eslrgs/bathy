# Loading Data

bathy supports multiple data sources and formats for bathymetric data.

## GEBCO OPeNDAP

You can download data directly from the GEBCO global bathymetry dataset via OPeNDAP:

```python
from bathy import Bathymetry

bath = Bathymetry.from_gebco_opendap(
    lon_range=(-12, -5),
    lat_range=(46, 50),
    year=2025,  # GEBCO version year
    save_path="data/region.nc",  # Optional: cache locally
)
```

### Caching behaviour

If `save_path` is provided and the file already exists, bathy will load from the file instead of downloading. This avoids redundant downloads.

### Preset regions

Common oceanographic regions are available as presets:

```python
from bathy import list_regions

# List all available regions
regions = list_regions()
print(regions[:10])
# ['antarctic', 'arabian_sea', 'arctic', 'baltic_sea', ...]

# Use a preset
bath = Bathymetry.from_gebco_opendap(region="mediterranean")
```

## Local NetCDF files

Load from local NetCDF files (e.g., downloaded GEBCO data):

```python
bath = Bathymetry(
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
bath = Bathymetry("path/to/bathymetry.tif")
```

GeoTIFF files are loaded using rioxarray with automatic coordinate renaming.

## Exporting data

### To GeoTIFF

```python
bath.to_geotiff("output.tif", crs="EPSG:4326")
```

## Data properties

After loading, you can inspect the data:

```python
print(bath.shape)       # Grid dimensions
print(bath.lon_range)   # Longitude bounds
print(bath.lat_range)   # Latitude bounds
print(bath)             # Summary representation
```
