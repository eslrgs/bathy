# 🌐 bathy

![Status](https://img.shields.io/badge/status-experimental-red)

Python package for exploring bathymetric grids.

## Installation

```bash
uv pip install .
```

## Basic usage

```python
from bathy import Bathymetry

# Load data
bath = Bathymetry('GEBCO_2023.nc', lon_range=(-10, 0), lat_range=(50, 60))

# Analyse
stats = bath.summary()
coverage = bath.coverage()

# Visualize
bath.plot_bathy()
bath.plot_slope()
bath.plot_depth_zones()

# Profiles
prof = bath.profile((-8, 52), (-2, 58))
prof.stats()
prof.plot()

# Canyon analysis
canyons = prof.get_canyons(prominence=100)
prof.plot_canyons(prominence=100)

# Multiple profiles
from bathy import profile

prof1 = bath.profile((-8, 52), (-2, 58), name="Profile 1")
prof2 = bath.profile((-8, 53), (-2, 59), name="Profile 2")
profile.plot_profiles([prof1, prof2])
profile.compare_stats([prof1, prof2])
```

## Features

**Bathymetry class:**
- Analysis: `summary()`, `depth_stats()`, `coverage()`, `slope()`, `aspect()`, `curvature()`
- Plotting: `plot_bathy()`, `plot_hillshade()`, `plot_slope()`, `plot_curvature()`, `plot_depth_zones()`, `plot_histogram()`, `plot_surface3d()`
- Profiles: `profile()`

**Profile class:**
- Analysis: `stats()`, `max_depth()`, `gradient()`, `concavity_index()`, `get_canyons()`
- Plotting: `plot()`, `plot_gradient()`, `plot_canyons()`
- Constructors: `from_coordinates()`, `cross_sections()`, `from_shapefile()`

**Multi-profile functions (profile module):**
- Analysis: `compare_stats()`
- Plotting: `plot_profiles()`, `plot_profiles_grid()`, `plot_profiles_map()`

## Preset regions

33 preset regions available:

```python
from bathy import list_regions, Bathymetry

list_regions()  # ['antarctic', 'arabian_sea', 'arctic', ...]

bath = Bathymetry('GEBCO_2023.nc', region='mediterranean')
```

## Examples

See [notebooks/basic_usage.ipynb](notebooks/basic_usage.ipynb) and [notebooks/profiles.ipynb](notebooks/profiles.ipynb).

## Development

```bash
git clone https://github.com/eslrgs/bathy.git
cd bathy
uv sync --all-groups
```

## License

MIT License - see [LICENSE](LICENSE) for details.
