# Visualisation

bathy provides a range of visualisation functions for bathymetric data. All `plot_*` functions return `(fig, ax)` so you can annotate, save, or compose figures.

## Elevation maps

### Basic bathymetry

```python
bathy.plot_bathy(data)
```

Uses the cmocean `deep` colormap (light = shallow, dark = deep).

### With contours

```python
# Specific contour depths
bathy.plot_bathy(data, contours=[-200, -1000, -2000, -4000])

# Number of contours
bathy.plot_bathy(data, contours=10)
```

### Custom colormap

```python
bathy.plot_bathy(data, cmap="viridis")
```

## Terrain analysis

### Hillshade

```python
bathy.plot_hillshade(data, azimuth=315, altitude=45)
```

### Slope

```python
bathy.plot_slope(data)

# Clip extreme values
bathy.plot_slope(data, vmax=20)  # Cap at 20 degrees
```

### Curvature

```python
bathy.plot_curvature(data)
```

Positive values indicate convex features (ridges), negative indicate concave features (valleys).

### Bathymetric Position Index (BPI)

```python
bathy.plot_bpi(data, radius_km=2.0)
```

BPI identifies ridges (positive) and valleys (negative) relative to the surrounding terrain. The `radius_km` parameter controls the neighbourhood size.

### Aspect

```python
bathy.plot_aspect(data)
```

Aspect is the downslope direction a surface faces (0° = north, 90° = east, 180° = south, 270° = west), following the standard GIS convention. Uses a circular colormap so north is consistent at both ends of the scale. Flat areas are shown as NaN.

### Rugosity

```python
bathy.plot_rugosity(data, radius_km=1.0)

# Clip extreme values
bathy.plot_rugosity(data, vmax=0.05)
```

Rugosity (Vector Ruggedness Measure) quantifies terrain complexity. Values range from 0 (flat) to 1 (maximally rough). Higher values indicate structurally complex seabed — useful for identifying hard substrate and reef habitat. The `radius_km` parameter controls the neighbourhood size.

### Geomorphons

```python
bathy.plot_geomorphons(data, lookup_km=2.0)

# Finer scale with tighter flatness threshold
bathy.plot_geomorphons(data, lookup_km=1.0, flatness_threshold=0.5)
```

Geomorphons classify terrain into 10 morphological forms (flat, peak, ridge, shoulder, spur, slope, hollow, footslope, valley, pit) by comparing each cell to eight neighbours at the lookup distance. Colours follow a warm (elevated) → grey (neutral) → cool (depressed) scheme. The `lookup_km` parameter controls the scale of analysis; larger values capture broader landscape forms.

### Overview (all maps)

```python
# All eight terrain analyses in one figure
bathy.plot_overview(data)

# Custom neighbourhood scales
bathy.plot_overview(data, bpi_radius_km=2.0, rugosity_radius_km=2.0, geomorphons_lookup_km=5.0)
```

Displays bathymetry, hillshade, slope, aspect, curvature, BPI, rugosity, and geomorphons in a single 4 × 2 figure. Useful for a quick scan of all key bathymetric characteristics.

## Depth zones

Classify bathymetry into depth zones:

```python
# Default zones: shelf, slope, abyss, deep
bathy.plot_depth_zones(data)

# Custom zones
bathy.plot_depth_zones(
    data,
    zones=[0, -200, -2000, -6000],
    labels=["Shelf", "Slope", "Abyss", "Hadal"],
)
```

## Statistical plots

### Histogram

```python
bathy.plot_histogram(data, bins=100)
```

### Hypsometric curve

```python
bathy.plot_hypsometric_curve(data)
```

## 3D surface

```python
bathy.plot_surface3d(
    data,
    stride=10,                 # Downsample factor
    vertical_exaggeration=50,  # Z-axis scaling
    smooth=5,                  # Optional smoothing
    elev=30,                   # View elevation
    azim=-60,                  # View azimuth
)
```

## Contours on any plot

All plot functions support the `contours` parameter:

```python
bathy.plot_hillshade(data, contours=[-200, -4000])
bathy.plot_slope(data, contours=5)
bathy.plot_depth_zones(data, contours=[-200, -1000])
```

## Saving figures

```python
fig, ax = bathy.plot_bathy(data)
fig.savefig("bathymetry.png", dpi=300, bbox_inches="tight")
```
