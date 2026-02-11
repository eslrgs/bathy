# bathy

**Lightweight Python package for exploring bathymetry data.**

bathy provides tools for loading, analysing, and visualising bathymetric (ocean depth) data. It integrates with GEBCO global bathymetry datasets and supports common geospatial formats.

## Features

- **Data loading**: Load from NetCDF, GeoTIFF, or download directly from GEBCO
- **Profile analysis**: Extract and analyse bathymetric profiles with canyon detection and knickpoint identification
- **Terrain analysis**: Calculate slope, curvature, and hypsometric indices
- **Visualisation**: Publication-ready plots including hillshade, depth zones, and 3D surfaces

## Quick example

```python
from bathy import Bathymetry

# Download data from GEBCO
bath = Bathymetry.from_gebco_opendap(
    lon_range=(-12, -5),
    lat_range=(46, 50),
    save_path="data/celtic_sea.nc",
)

# Visualise
bath.plot_bathy()

# Extract a profile
profile = bath.profile(start=(-11, 48), end=(-6, 48))
profile.plot()
```

## Installation

```bash
git clone https://github.com/eslrgs/bathy.git
cd bathy
uv sync
```

See the [Installation guide](getting-started/installation.md) for more details.
