# Changelog

## 0.1.0 — 2026-03-11

Initial release.

### Features

- **Data loading**: Load bathymetry from NetCDF, GeoTIFF, or GEBCO OPeNDAP with 28 preset regions
- **Grid analysis**: Slope, curvature, aspect, BPI, rugosity, geomorphons, hypsometric index/curve, summary statistics
- **Profiles**: Extract straight-line or multi-waypoint profiles, cross-sections, import from vector files (Shapefile, GeoPackage, etc.)
- **Profile analysis**: Stats, gradient, concavity index, knickpoint detection, canyon identification
- **Visualisation**: 13 plot functions for grids (bathymetry, hillshade, slope, aspect, curvature, BPI, rugosity, geomorphons, depth zones, histogram, 3D surface, hypsometric curve, overview panel) and 7 for profiles
- **Interactive drawing**: PyQt6 app for drawing profiles on a map (`bathy-draw` CLI)
- **Export**: Profiles to GeoPackage/Shapefile via GeoDataFrame; grids to GeoTIFF
- **Sample data**: Built-in sample dataset for quick exploration
- Supports geographic (lon/lat) and projected (x/y) coordinate systems
