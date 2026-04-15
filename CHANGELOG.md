# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.3.0] - 2026-04-15

### Added
- **Volume & area**: `volume` and `area` calculations 
- `hypsometric_curve` gains `absolute=True` option for depth-area curves in real units
- **Grid operations**: new `grid` module with `clip`, `resample`, `reproject`, `merge`, `fill_gaps`
- `resample` supports both `resolution_degrees` and `resolution_m` parameters
- `clip` supports region presets and vector file paths in addition to GeoDataFrames
- `merge` handles overlapping grids with configurable strategy (mean, min, max, first)
- Grid operations notebook example (`examples/grid_operations.ipynb`)
- API docs page for grid operations
- `ipywidgets` added to notebook optional dependencies

## [0.2.0] - 2026-04-01

### Added
- `load_emodnet_wcs` for loading data from EMODnet WCS
- `smooth` method for grid smoothing
- `plot_interactive` for interactive Folium maps
- `contours` extraction method
- Download size estimation and logging for GEBCO and EMODnet downloads
- Large download guard — `save_path` required for regions over 500 MB
- GEBCO year validation (valid years: 2019-2025)
- Automatic temp file cleanup when `save_path` is not provided
- CD workflow for PyPI publishing
- Smoke test for release pipeline

### Fixed
- Canyon width detection accuracy

## [0.1.1] - 2026-03-11

### Added
- Sample dataset support

## [0.1.0] - 2026-03-11

Initial release.

### Added
- **IO**: `load_bathymetry`, `load_gebco_opendap`, `to_geotiff`, `list_regions` with 28 preset regions
- **Analysis**: `slope`, `aspect`, `curvature`, `rugosity`, `bpi`, `geomorphons`, `hypsometric_index`, `hypsometric_curve`, `summary`
- **Plotting**: `plot_bathy`, `plot_hillshade`, `plot_slope`, `plot_aspect`, `plot_curvature`, `plot_bpi`, `plot_rugosity`, `plot_geomorphons`, `plot_overview`, `plot_depth_zones`, `plot_histogram`, `plot_surface3d`, `plot_hypsometric_curve`
- **Profiles**: `extract_profile`, `profile_from_coordinates`, `cross_sections`, `profiles_from_file`, `profiles_from_gdf`
- **Profile analysis**: `profile_stats`, `max_depth`, `gradient`, `concavity_index`, `knickpoints`, `get_canyons`, `compare_stats`, `to_gdf`
- **Profile plotting**: `plot_profile`, `plot_profiles`, `plot_profiles_grid`, `plot_profiles_map`, `plot_gradient`, `plot_knickpoints`, `plot_canyons`
- **Draw**: PyQt6 interactive profile drawing app with waypoint editing, save/load as GeoPackage
- Projected CRS support
- Interpolation method parameter for profile extraction
- Input validation across public API
- CI with Python 3.12 and 3.13
- MkDocs documentation site
