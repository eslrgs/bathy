# Profiles

Bathymetric profiles allow you to analyse depth variations along a transect.

## Creating profiles

### From Bathymetry object

```python
from bathy import Bathymetry

bath = Bathymetry.from_gebco_opendap(region="mediterranean")

# Create a profile with specific number of points
profile = bath.profile(
    start=(-5, 36),
    end=(10, 40),
    num_points=200,
    name="Western Mediterranean",
)

# Or with specific point spacing (in km)
profile = bath.profile(
    start=(-5, 36),
    end=(10, 40),
    point_spacing=5.0,  # 5 km spacing
)
```

### From shapefile or GeoDataFrame

Load profiles from a shapefile or an in-memory GeoDataFrame of LineStrings:

```python
from bathy import Profile

# From shapefile
profiles = Profile.from_shapefile(
    bath.data,
    "path/to/canyons.shp",
    id_column="NAME",
)

# From GeoDataFrame
import geopandas as gpd

gdf = gpd.read_file("canyons.gpkg")
profiles = Profile.from_gdf(bath.data, gdf, id_column="NAME")
```

Features outside the DEM extent are skipped automatically. Any non-geometry columns in the GeoDataFrame are stored as profile metadata.

## Profile analysis

### Basic statistics

```python
profile.stats()
```

### Maximum depth

```python
distance, depth = profile.max_depth()
print(f"Max depth: {depth:.0f} m at {distance:.1f} km")
```

### Gradient

```python
gradient = profile.gradient()  # Returns numpy array
```

## Canyon detection

Identify submarine canyons along a profile:

```python
canyons = profile.get_canyons(prominence=500)  # Minimum 500m prominence
print(canyons)
```

Returns a DataFrame with canyon properties including floor depth, width, and cross-sectional area.

## Knickpoint detection

Identify knickpoints (abrupt slope changes):

```python
# Auto-threshold (mean + 2 std)
kp = profile.knickpoints()

# Custom threshold
kp = profile.knickpoints(threshold=5)

# With smoothing
kp = profile.knickpoints(smooth=3)
```

## Visualisation

### Basic profile plot

```python
profile.plot()
```

### With smoothing

```python
profile.plot(smooth=3.0)
```

### With canyons highlighted

```python
profile.plot_canyons(prominence=500)
```

### With knickpoints

```python
kp = profile.knickpoints()
profile.plot_knickpoints(kp)
```

## Cross-sections

Create perpendicular cross-sections along a main profile:

```python
from bathy import Profile

# Main profile
main = bath.profile((-11, 47.5), (-6.5, 49), point_spacing=1.0)

# Create cross-sections every 20 km, 30 km wide
cross_sections = Profile.cross_sections(
    bath.data,
    main,
    interval_km=20,
    section_width_km=30,
    num_points=50,
)
```

## GeoDataFrame export

Export a single profile or a collection to a GeoDataFrame for use with GeoPandas, QGIS, or spatial analysis workflows:

```python
from bathy.profile import to_gdf

# Single profile
gdf = to_gdf(prof)

# Multiple profiles
gdf = to_gdf([prof1, prof2, prof3])
gdf.to_file("profiles.gpkg", driver="GPKG")
```

Each row contains the profile geometry (LineString), summary statistics (`total_distance_km`, `min_elevation_m`, `max_elevation_m`, `mean_elevation_m`), and any scalar metadata attached to the profile.

## Comparing multiple profiles

```python
from bathy import profile

# Create several profiles
profiles = [
    bath.profile((-11, lat), (-6.5, lat), name=f"{lat}N")
    for lat in [46.5, 47.5, 48.5]
]

# Compare statistics
profile.compare_stats(profiles)

# Plot together
profile.plot_profiles(profiles)

# Show on map
profile.plot_profiles_map(profiles)
```
