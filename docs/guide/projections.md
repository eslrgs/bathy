# Projected Coordinate Systems

bathy automatically detects the coordinate reference system (CRS) of your data and adapts its behaviour accordingly. Both geographic (longitude/latitude) and projected (e.g. UTM) coordinate systems are supported.

## How it works

When you load data via `load_bathymetry`, bathy reads the CRS metadata from the file:

- **Geographic CRS** (e.g. EPSG:4326): coordinates are longitude/latitude in degrees. Distances are computed geodesically on the WGS84 ellipsoid.
- **Projected CRS** (e.g. UTM zones): coordinates are eastings/northings in metres. Distances are computed using Euclidean geometry.
- **No CRS**: treated as geographic for backward compatibility.

CRS detection uses [rioxarray](https://corteva.github.io/rioxarray/), which reads CRS metadata from GeoTIFF tags, NetCDF spatial_ref variables, and other standard formats.

## Loading projected data

GeoTIFF files with a projected CRS are loaded with their native coordinates preserved:

```python
import bathy

# UTM GeoTIFF — coordinates stay as x/y in metres
data = bathy.load_bathymetry("multibeam_utm29n.tif")

print(data.dims)   # ('y', 'x')
print(data.x[:3])  # e.g. [500000, 500050, 500100]
```

Geographic GeoTIFFs are renamed to `lon`/`lat` as usual:

```python
data = bathy.load_bathymetry("gebco_subset.tif")

print(data.dims)  # ('lat', 'lon')
```

## What changes with projected data

All functions adapt automatically — no extra parameters needed.

### Analysis

Cell sizes and gradients are computed directly from coordinate spacing (no geodesic computation). This is both faster and more appropriate for projected grids:

```python
sl = bathy.slope(data)       # Cell size from x/y spacing
cu = bathy.curvature(data)   # Same
bp = bathy.bpi(data)         # Same
```

### Profiles

Profile distances use Euclidean geometry. Coordinates are passed as `(x, y)` tuples:

```python
prof = bathy.extract_profile(
    data,
    start=(502000, 5540000),
    end=(508000, 5545000),
    point_spacing=100.0,
)

# Distances are in metres (Euclidean)
print(prof.distances[-1])  # ~7810 m
```

Cross-sections use trigonometric perpendicular offsets instead of geodesic bearings:

```python
sections = bathy.cross_sections(
    data, prof,
    interval_m=2000,
    section_width_m=1000,
)
```

### Plotting

Axis labels update automatically:

- Geographic: "Longitude (°)" / "Latitude (°)"
- Projected: "Easting (m)" / "Northing (m)"

The 3D surface plot skips the `cos(lat)` aspect correction for projected data, since eastings and northings already share the same unit.

### GeoDataFrame export

Exported profiles carry the data's CRS:

```python
gdf = bathy.to_gdf(prof)
print(gdf.crs)  # EPSG:32629
```

When loading profiles from a GeoDataFrame or file, the GDF's CRS must match the data's CRS. For geographic data, the GDF is reprojected to EPSG:4326 automatically.

## Limitations

- **No on-the-fly reprojection.** If your data is in UTM zone 29N, profile coordinates must also be in UTM zone 29N.
- **Metre-based projected CRS only.** The package assumes projected coordinates are in metres (as with UTM, British National Grid, etc.). Non-metre projections (e.g. US State Plane in feet) are not explicitly supported.
- **Mixed CRS is not supported.** All data and profiles in a workflow should share the same CRS.
