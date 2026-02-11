# Quick Start

This guide covers the basics of loading bathymetry data and creating visualisations.

## Loading data

### From GEBCO

The easiest way to get started is to download data directly from the GEBCO OPeNDAP server:

```python
from bathy import Bathymetry

bath = Bathymetry.from_gebco_opendap(
    lon_range=(-12, -5),
    lat_range=(46, 50),
    save_path="data/my_region.nc",  # Optional: save for reuse
)
```

If `save_path` exists, the download is skipped and data is loaded from the file.

### Using preset regions

For convenience, common oceanographic regions are available:

```python
from bathy import Bathymetry, list_regions

# See available regions
print(list_regions())

# Load a preset region
bath = Bathymetry.from_gebco_opendap(region="mediterranean")
```

### From local files

Load from NetCDF or GeoTIFF:

```python
# NetCDF
bath = Bathymetry("path/to/gebco.nc", lon_range=(-10, -5), lat_range=(50, 55))

# GeoTIFF
bath = Bathymetry("path/to/bathymetry.tif")
```

## Basic visualisation

```python
# Elevation map
bath.plot_bathy()

# With contours
bath.plot_bathy(contours=[-200, -1000, -2000, -4000])

# Hillshade
bath.plot_hillshade()

# Slope map
bath.plot_slope()

# Depth zones
bath.plot_depth_zones()
```

## Creating profiles

Extract a bathymetric profile between two points:

```python
# Create profile with 1 km point spacing
profile = bath.profile(
    start=(-11, 48),
    end=(-6, 48),
    point_spacing=1.0,
    name="East-West Profile",
)

# Plot the profile
profile.plot()

# Get statistics
profile.stats()
```

## Summary statistics

```python
# Overall statistics
bath.summary()

# Depth statistics (underwater only)
bath.depth_stats()

# Land/sea coverage
bath.coverage()

# Hypsometric index
hi = bath.hypsometric_index()
print(f"Hypsometric Index: {hi:.3f}")
```
