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
