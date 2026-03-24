# 🌐 bathy

![Status](https://img.shields.io/badge/status-experimental-red)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://eslrgs.github.io/bathy)

Python package for exploring bathymetric grids.

**[Documentation](https://eslrgs.github.io/bathy)** · [Installation](#installation) · [Examples](#examples)

## Basic usage

```python
import bathy

# Load from file
data = bathy.load_bathymetry("GEBCO_2025.nc", lon_range=(-10, 0), lat_range=(50, 60))

# Or download GEBCO data via OPeNDAP
data = bathy.load_gebco_opendap(lon_range=(-10, 0), lat_range=(50, 60))

# Analyse
bathy.summary(data)
bathy.slope(data)

# Visualise
bathy.plot_bathy(data)
bathy.plot_slope(data)
bathy.plot_depth_zones(data)

# Profiles
prof = bathy.extract_profile(data, (-8, 52), (-2, 58), name="Celtic Sea")
bathy.profile_stats(prof)
bathy.plot_profile(prof)

# Canyon analysis
canyons = bathy.get_canyons(prof, prominence=100)
bathy.plot_canyons(prof, canyons)

# Multiple profiles
profiles = [
    bathy.extract_profile(data, (-8, lat), (-2, lat + 6), name=f"{lat}N")
    for lat in [52, 53, 54]
]
bathy.plot_profiles(profiles)
bathy.compare_stats(profiles)
```

## Installation

```bash
uv pip install .
```

## Features

| Category | Functions |
|---|---|
| **IO** | `load_bathymetry`, `load_gebco_opendap`, `load_emodnet_wcs`, `to_geotiff`, `list_regions` |
| **Bathymetric analysis** | `slope`, `aspect`, `curvature`, `rugosity`, `bpi`, `geomorphons`, `contours`, `smooth`, `hypsometric_index`, `hypsometric_curve`, `summary` |
| **Grid plotting** | `plot_bathy`, `plot_hillshade`, `plot_slope`, `plot_aspect`, `plot_curvature`, `plot_bpi`, `plot_rugosity`, `plot_geomorphons`, `plot_overview`, `plot_depth_zones`, `plot_histogram`, `plot_surface3d`, `plot_hypsometric_curve`, `plot_interactive` |
| **Profiles** | `extract_profile`, `profile_from_coordinates`, `cross_sections`, `profiles_from_file`, `profiles_from_gdf` |
| **Profile analysis** | `profile_stats`, `max_depth`, `gradient`, `concavity_index`, `knickpoints`, `get_canyons`, `compare_stats`, `to_gdf` |
| **Profile plotting** | `plot_profile`, `plot_profiles`, `plot_profiles_grid`, `plot_profiles_map`, `plot_gradient`, `plot_knickpoints`, `plot_canyons` |
| **Draw** | `draw_profile` — draw and edit profiles on a map with drag, undo, delete, and insert waypoints (PyQt6 desktop window) |

## Preset regions

28 preset regions available:

```python
import bathy

bathy.list_regions()  # ['arabian_sea', 'baltic_sea', 'bay_of_bengal', ...]

data = bathy.load_gebco_opendap(region="mediterranean")
```

## Examples

See [examples/basic_usage.ipynb](examples/basic_usage.ipynb), [examples/profiles.ipynb](examples/profiles.ipynb), and [examples/draw_profile.py](examples/draw_profile.py).

### Profile drawing (desktop app)

Draw profiles interactively in a PyQt6 window. Requires `uv pip install bathy[draw]`.

```bash
uv run bathy-draw path/to/data.nc
```

Or from Python:

```python
import bathy

data = bathy.load_bathymetry("data.nc")
profiles = bathy.draw_profile(data)
```

- Left-click to add waypoints, right-click to finish a profile, double-click to stop
- Drag waypoints to reposition, press **z** to undo, shift-click to delete
- Save/load profiles as GeoPackage, toggle visibility per profile

## Development

```bash
git clone https://github.com/eslrgs/bathy.git
cd bathy
uv sync
pre-commit install
```

```bash
just format  # Format and lint
just test    # Run tests
```

## License

MIT License - see [LICENSE](LICENSE) for details.
