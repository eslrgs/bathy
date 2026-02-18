# Visualisation

bathy provides a range of visualisation methods for bathymetric data.

## Elevation maps

### Basic bathymetry

```python
bath.plot_bathy()
```

Uses the cmocean `deep` colormap (light = shallow, dark = deep).

### With contours

```python
# Specific contour depths
bath.plot_bathy(contours=[-200, -1000, -2000, -4000])

# Number of contours
bath.plot_bathy(contours=10)
```

### Custom colormap

```python
bath.plot_bathy(cmap="viridis")
```

## Terrain analysis

### Hillshade

```python
bath.plot_hillshade(azimuth=315, altitude=45)
```

### Slope

```python
bath.plot_slope()

# Clip extreme values
bath.plot_slope(vmax=20)  # Cap at 20 degrees
```

### Curvature

```python
bath.plot_curvature()
```

Positive values indicate convex features (ridges), negative indicate concave features (valleys).

### Bathymetric Position Index (BPI)

```python
bath.plot_bpi(radius_km=2.0)
```

BPI identifies ridges (positive) and valleys (negative) relative to the surrounding terrain. The `radius_km` parameter controls the neighbourhood size.

### Aspect

```python
bath.plot_aspect()
```

Aspect is the compass direction of the steepest upslope gradient (0° = north, 90° = east, 180° = south, 270° = west). Uses a circular colormap so north is consistent at both ends of the scale. Flat areas are shown as NaN.

### Rugosity

```python
bath.plot_rugosity(radius_km=1.0)

# Clip extreme values
bath.plot_rugosity(vmax=0.05)
```

Rugosity (Vector Ruggedness Measure) quantifies terrain complexity. Values range from 0 (flat) to 1 (maximally rough). Higher values indicate structurally complex seabed — useful for identifying hard substrate and reef habitat. The `radius_km` parameter controls the neighbourhood size.

### Geomorphons

```python
bath.plot_geomorphons(lookup_km=2.0)

# Finer scale with tighter flatness threshold
bath.plot_geomorphons(lookup_km=1.0, flatness_threshold=0.5)
```

Geomorphons classify terrain into 10 morphological forms (flat, peak, ridge, shoulder, spur, slope, hollow, footslope, valley, pit) by comparing each cell to eight neighbours at the lookup distance. Colours follow a warm (elevated) → grey (neutral) → cool (depressed) scheme. The `lookup_km` parameter controls the scale of analysis; larger values capture broader landscape forms.

## Depth zones

Classify bathymetry into depth zones:

```python
# Default zones: shelf, slope, abyss, deep
bath.plot_depth_zones()

# Custom zones
bath.plot_depth_zones(
    zones=[0, -200, -2000, -6000],
    labels=["Shelf", "Slope", "Abyss", "Hadal"],
)
```

## Statistical plots

### Histogram

```python
bath.plot_histogram(bins=100)
```

### Hypsometric curve

```python
bath.plot_hypsometric_curve()
```

## 3D surface

```python
bath.plot_surface3d(
    stride=10,                 # Downsample factor
    vertical_exaggeration=50,  # Z-axis scaling
    smooth=5,                  # Optional smoothing
    elev=30,                   # View elevation
    azim=-60,                  # View azimuth
)
```

## Contours on any plot

All plot methods support the `contours` parameter:

```python
bath.plot_hillshade(contours=[-200, -4000])
bath.plot_slope(contours=5)
bath.plot_depth_zones(contours=[-200, -1000])
```
