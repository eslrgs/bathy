# Quick Start

This guide covers the basics of loading bathymetry data and creating visualisations.

## Loading data

### From remote sources

The easiest way to get started is to download data directly. Several global and regional sources are available:

```python
import bathy

# GEBCO global (~450 m / 15 arc-second)
data = bathy.load_gebco_opendap(
    lon_range=(-12, -5),
    lat_range=(46, 50),
    save_path="data/my_region.nc",  # Optional: save for reuse
)

# ETOPO 2022 global (60s / 30s / 15s resolution)
data = bathy.load_etopo(lon_range=(-12, -5), lat_range=(46, 50), resolution="60s")

# EMODnet (~115 m, European seas only)
data = bathy.load_emodnet_wcs(lon_range=(-10, -5), lat_range=(50, 55))

# NOAA Coastal Relief Model (~90 m, US coasts only)
data = bathy.load_noaa_crm(lon_range=(-72, -70), lat_range=(41, 43))
```

If `save_path` is provided and the file already exists, the download is skipped and data is loaded from the file.

### Using preset regions

For convenience, common oceanographic regions are available:

```python
import bathy

# See available regions
print(bathy.list_regions())

# Load a preset region — works with any load function
data = bathy.load_gebco_opendap(region="mediterranean")
```

### From local files

Load from NetCDF or GeoTIFF:

```python
# NetCDF
data = bathy.load_bathymetry("path/to/gebco.nc", lon_range=(-10, -5), lat_range=(50, 55))

# GeoTIFF
data = bathy.load_bathymetry("path/to/bathymetry.tif")
```

## Basic visualisation

```python
# Elevation map
bathy.plot_bathy(data)

# With contours
bathy.plot_bathy(data, contours=[-200, -1000, -2000, -4000])

# Hillshade
bathy.plot_hillshade(data)

# Slope map
bathy.plot_slope(data)

# Depth zones
bathy.plot_depth_zones(data)
```

All plot functions return `(fig, ax)` so you can annotate or save:

```python
fig, ax = bathy.plot_bathy(data)
ax.set_title("Celtic Sea")
fig.savefig("celtic_sea.png", dpi=300)
```

## Creating profiles

Extract a bathymetric profile between two points:

```python
# Create profile with 1 km point spacing
prof = bathy.extract_profile(
    data,
    start=(-11, 48),
    end=(-6, 48),
    point_spacing=1000.0,
    name="East-West Profile",
)

# Plot the profile
bathy.plot_profile(prof)

# Get statistics
bathy.profile_stats(prof)
```

## Smoothing

Gaussian smooth a grid to suppress noise before derived analysis:

```python
smoothed = bathy.smooth(data, sigma_km=2.0)

# Chain with other analysis
bathy.plot_slope(smoothed)
```

## Contour extraction

Extract depth contours as vector geometries for export or spatial analysis:

```python
# Contours at specific depths
gdf = bathy.contours(data, levels=[-200, -1000, -2000])

# Regular interval
gdf = bathy.contours(data, interval=500)

# Export to file
gdf.to_file("contours.gpkg")
```

## Interactive map

Explore bathymetry on an interactive Leaflet basemap with toggleable analysis overlays:

```python
# Basic interactive map
bathy.plot_interactive(data)

# With analysis overlays
bathy.plot_interactive(data, overlays={
    "Slope": bathy.slope(data),
    "Rugosity": bathy.rugosity(data),
})
```

## Summary statistics

```python
# Overall statistics (delegates to xarray describe)
bathy.summary(data)

# Hypsometric index
hi = bathy.hypsometric_index(data)
print(f"Hypsometric Index: {hi:.3f}")
```
