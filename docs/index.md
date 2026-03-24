# bathy

**Lightweight Python package for exploring bathymetry data.**

bathy provides tools for loading, analysing, and visualising bathymetric (ocean depth) data. It integrates with GEBCO global bathymetry datasets and supports common geospatial formats.

## Features

- **Data loading**: Load from NetCDF, GeoTIFF, or download directly from GEBCO and EMODnet. Supports both geographic (lon/lat) and projected (e.g. UTM) coordinate systems
- **Profile analysis**: Extract and analyse bathymetric profiles with canyon detection and knickpoint identification
- **Interactive drawing**: Draw and edit profiles on a map with drag, undo, delete, and insert
- **Bathymetric analysis**: Calculate slope, curvature, BPI, rugosity, hypsometric indices, contour extraction, and Gaussian smoothing
- **Visualisation**: Publication-ready plots including hillshade, depth zones, 3D surfaces, and interactive Leaflet maps

## Quick example

```python
import bathy

# Download data from GEBCO
data = bathy.load_gebco_opendap(
    lon_range=(-12, -5),
    lat_range=(46, 50),
    save_path="data/celtic_sea.nc",
)

# Visualise
bathy.plot_bathy(data)

# Extract a profile
prof = bathy.extract_profile(data, start=(-11, 48), end=(-6, 48))
bathy.plot_profile(prof)
```

## Installation

```bash
git clone https://github.com/eslrgs/bathy.git
cd bathy
uv sync
```

See the [Installation guide](getting-started/installation.md) for more details.
