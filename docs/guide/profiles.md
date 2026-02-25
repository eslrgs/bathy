# Profiles

Bathymetric profiles allow you to analyse depth variations along a transect.

## Interactive drawing

Draw profiles by clicking directly on the map (requires `%matplotlib widget` in Jupyter):

```python
%matplotlib widget
import bathy

data = bathy.load_gebco_opendap(lon_range=(-12, -4), lat_range=(50, 56))
result = bathy.draw_profile(data)
# Left-click to add waypoints, right-click to finish each profile
# Double-click to stop drawing
result["profiles"]  # list of drawn Profile objects
```

## Creating profiles

### From coordinates

```python
import bathy

data = bathy.load_gebco_opendap(region="mediterranean")

# Create a profile with specific number of points
prof = bathy.extract_profile(
    data,
    start=(-5, 36),
    end=(10, 40),
    num_points=200,
    name="Western Mediterranean",
)

# Or with specific point spacing (in km)
prof = bathy.extract_profile(
    data,
    start=(-5, 36),
    end=(10, 40),
    point_spacing=5.0,  # 5 km spacing
)
```

### From shapefile or GeoDataFrame

Load profiles from a shapefile or an in-memory GeoDataFrame of LineStrings:

```python
# From shapefile
profiles = bathy.profiles_from_shapefile(
    data,
    "path/to/canyons.shp",
    id_column="NAME",
)

# From GeoDataFrame
import geopandas as gpd

gdf = gpd.read_file("canyons.gpkg")
profiles = bathy.profiles_from_gdf(data, gdf, id_column="NAME")
```

Features outside the DEM extent are skipped automatically. Any non-geometry columns in the GeoDataFrame are stored as profile metadata.

## Profile analysis

### Basic statistics

```python
bathy.profile_stats(prof)
```

### Maximum depth

```python
distance, depth = bathy.max_depth(prof)
print(f"Max depth: {depth:.0f} m at {distance:.1f} km")
```

### Gradient

```python
grad = bathy.gradient(prof)  # Returns numpy array
```

## Canyon detection

Identify submarine canyons along a profile:

```python
canyons = bathy.get_canyons(prof, prominence=500)  # Minimum 500m prominence
print(canyons)
```

Returns a DataFrame with canyon properties including floor depth, width, and cross-sectional area.

## Knickpoint detection

Identify knickpoints (abrupt slope changes):

```python
# Auto-threshold (mean + 2 std)
kp = bathy.knickpoints(prof)

# Custom threshold
kp = bathy.knickpoints(prof, threshold=5)

# With smoothing
kp = bathy.knickpoints(prof, smooth=3)
```

## Visualisation

### Basic profile plot

```python
bathy.plot_profile(prof)
```

### With smoothing

```python
bathy.plot_profile(prof, smooth=3.0)
```

### With canyons highlighted

```python
bathy.plot_canyons(prof, prominence=500)
```

### With knickpoints

```python
bathy.plot_knickpoints(prof)

# Or with pre-computed knickpoints
kp = bathy.knickpoints(prof)
bathy.plot_knickpoints(prof, kp)
```

All profile plot functions return `(fig, ax)`.

## Cross-sections

Create perpendicular cross-sections along a main profile:

```python
# Main profile
main = bathy.extract_profile(data, (-11, 47.5), (-6.5, 49), point_spacing=1.0)

# Create cross-sections every 20 km, 30 km wide
x_sections = bathy.cross_sections(
    data,
    main,
    interval_km=20,
    section_width_km=30,
    num_points=50,
)

# Plot on map
bathy.plot_profiles_map(x_sections, bathymetry_data=data, main_profile=main)

# Plot in grid
bathy.plot_profiles_grid(x_sections[:6], cols=3, main_profile=main)
```

## GeoDataFrame export

Export a single profile or a collection to a GeoDataFrame for use with GeoPandas, QGIS, or spatial analysis workflows:

```python
# Single profile
gdf = bathy.to_gdf(prof)

# Multiple profiles
gdf = bathy.to_gdf([prof1, prof2, prof3])
gdf.to_file("profiles.gpkg", driver="GPKG")
```

Each row contains the profile geometry (LineString), summary statistics (`total_distance_km`, `min_elevation_m`, `max_elevation_m`, `mean_elevation_m`), and any scalar metadata attached to the profile.

## Comparing multiple profiles

```python
# Create several profiles
profiles = [
    bathy.extract_profile(data, (-11, lat), (-6.5, lat), name=f"{lat}N")
    for lat in [46.5, 47.5, 48.5]
]

# Compare statistics
bathy.compare_stats(profiles)

# Plot together
bathy.plot_profiles(profiles)

# Show on map
bathy.plot_profiles_map(profiles, bathymetry_data=data)
```
