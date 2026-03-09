# Profiles

Bathymetric profiles allow you to analyse depth variations along a transect.

## Drawing profiles

`draw_profile` opens a desktop window where you can draw and edit profiles directly on the bathymetry map. Install with `uv pip install bathy[draw]`.

### From the command line

```bash
uv run bathy-draw path/to/data.nc
```

### From Python

```python
import bathy

data = bathy.load_gebco_opendap(lon_range=(-12, -4), lat_range=(50, 56))
profiles = bathy.draw_profile(data)
```

Returns a `list[Profile]` when the window is closed.

### Controls

**Drawing:**

- **Left-click** to add waypoints along a profile path
- **Right-click** to finish the current profile and start a new one
- **Double-click** or press **Done** to stop drawing

**Editing:**

- **Drag** any waypoint to reposition it (works on finished profiles too)
- **Middle-click** or press **z** to undo the last placed waypoint
- **Shift-click** on a waypoint to delete it
- **Click on a line segment** to insert a new waypoint between two existing ones

**Window features:**

- Pan/zoom toolbar for navigating the map
- Coordinate readout as you move the cursor
- Profile list with visibility checkboxes (All/None toggle)
- Save and Load buttons for GeoPackage, Shapefile, or GeoJSON

### Saving and reloading

Profiles can be saved to a file and reloaded for further editing, like a GIS project:

```python
# Save drawn profiles
bathy.to_gdf(profiles).to_file("profiles.gpkg")

# Later: reload and continue editing
reloaded = bathy.profiles_from_file(data, "profiles.gpkg")
profiles = bathy.draw_profile(data, profiles=reloaded)
```

You can also use the **Save** and **Load** buttons in the window directly.

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

# Or with specific point spacing (in metres)
prof = bathy.extract_profile(
    data,
    start=(-5, 36),
    end=(10, 40),
    point_spacing=5000.0,  # 5 km spacing
)
```

### From shapefile or GeoDataFrame

Load profiles from a vector file or an in-memory GeoDataFrame of LineStrings:

```python
# From shapefile
profiles = bathy.profiles_from_file(
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
main = bathy.extract_profile(data, (-11, 47.5), (-6.5, 49), point_spacing=1000.0)

# Create cross-sections every 20 km, 30 km wide
x_sections = bathy.cross_sections(
    data,
    main,
    interval_m=20000,
    section_width_m=30000,
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
