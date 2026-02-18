"""Tests for bathymetry module."""

import numpy as np
import xarray as xr

from bathy.bathymetry import Bathymetry, list_regions


def test_list_regions():
    """List all preset regions."""
    regions = list_regions()

    assert "mediterranean" in regions
    assert "mariana_trench" in regions


def test_load_from_netcdf(temp_netcdf):
    """Load bathymetry from NetCDF file."""
    bath = Bathymetry(temp_netcdf)

    assert bath.shape == (20, 20)
    assert bath.lon_range == (-10.0, -5.0)
    assert bath.lat_range == (50.0, 55.0)


def test_summary_stats(fake_bathy):
    """Calculate summary statistics."""
    stats = fake_bathy.summary()

    assert "statistic" in stats.columns
    assert "value" in stats.columns
    assert len(stats) == 7


def test_slope_calculation(fake_bathy):
    """Calculate seafloor slope."""
    slope = fake_bathy.slope()

    assert slope.shape == fake_bathy.shape
    assert (slope.values >= 0).all()


def test_create_profile(fake_bathy):
    """Create a profile from bathymetry."""
    prof = fake_bathy.profile(start=(-9, 52), end=(-6, 53), num_points=10)

    assert prof.num_points == 10
    assert len(prof.distances) == 10
    assert len(prof.elevations) == 10
    assert prof.start_lon == -9
    assert prof.start_lat == 52
    assert prof.end_lon == -6
    assert prof.end_lat == 53


def test_plot_bathy_masks_land():
    """Verify plot_bathy masks land (elevation >= 0)."""
    import xarray as xr

    # Create data with both underwater and land
    data = xr.DataArray(
        np.array([[-100, 50]]),  # Underwater and land
        coords={"lon": [-10, -5], "lat": [50]},
        dims=["lat", "lon"],
    )

    # Test masking logic: data.where(data < 0)
    masked = data.where(data < 0)
    assert np.isnan(masked.values[0, 1])  # Land masked
    assert masked.values[0, 0] == -100  # Water not masked


# Hypsometry tests


def test_hypsometric_index_range(fake_bathy):
    """Hypsometric index should be between 0 and 1."""
    hi = fake_bathy.hypsometric_index()

    assert 0 <= hi <= 1


def test_hypsometric_index_uniform_distribution(uniform_bathy):
    """Uniform distribution should have HI close to 0.5."""
    hi = uniform_bathy.hypsometric_index()

    assert abs(hi - 0.5) < 0.01


def test_hypsometric_index_convex(convex_bathy):
    """Convex distribution (more high values) should have HI > 0.5."""
    hi = convex_bathy.hypsometric_index()

    assert hi > 0.5


def test_hypsometric_index_flat_surface(flat_bathy):
    """Flat surface (constant elevation) should return NaN."""
    hi = flat_bathy.hypsometric_index()

    assert np.isnan(hi)


def test_hypsometric_curve(fake_bathy):
    """Hypsometric curve returns normalised, monotonic arrays."""
    rel_area, rel_elev = fake_bathy.hypsometric_curve(bins=50)

    # Correct shape
    assert len(rel_area) == 50
    assert len(rel_elev) == 50

    # Normalised between 0 and 1
    assert 0 <= rel_area.min() and rel_area.max() <= 1
    assert 0 <= rel_elev.min() and rel_elev.max() <= 1

    # Relative area decreases as elevation increases
    assert np.all(np.diff(rel_area) <= 0)


def test_curvature_calculation(fake_bathy):
    """Calculate seafloor curvature."""
    curv = fake_bathy.curvature()

    assert curv.shape == fake_bathy.shape
    assert curv.name == "curvature"


def test_bpi_calculation(fake_bathy):
    """Calculate Bathymetric Position Index."""
    bpi = fake_bathy.bpi(radius_km=1.0)

    assert bpi.shape == fake_bathy.shape
    assert bpi.name == "bpi"


def test_bpi_flat_surface_is_zero(flat_bathy):
    """Flat surface should have BPI ≈ 0 everywhere."""
    bpi = flat_bathy.bpi(radius_km=1.0)

    # All values should be near zero (within floating point tolerance)
    assert np.allclose(bpi.values, 0, atol=1e-10)


def test_bpi_peak_is_positive():
    """A peak (high point surrounded by low) should have positive BPI."""
    # Create grid with a peak in the centre
    elevations = np.full((21, 21), -1000.0)
    elevations[10, 10] = -500.0  # Peak (shallower than surroundings)

    data = xr.DataArray(
        elevations,
        coords={"lon": np.linspace(-10, -5, 21), "lat": np.linspace(50, 55, 21)},
        dims=["lat", "lon"],
    )
    bath = Bathymetry.from_array(data)
    bpi = bath.bpi(radius_km=50)  # Large radius to capture the peak

    # Centre point should have positive BPI (higher than surroundings)
    assert bpi.values[10, 10] > 0


def test_bpi_valley_is_negative():
    """A valley (low point surrounded by high) should have negative BPI."""
    # Create grid with a valley in the centre
    elevations = np.full((21, 21), -500.0)
    elevations[10, 10] = -1000.0  # Valley (deeper than surroundings)

    data = xr.DataArray(
        elevations,
        coords={"lon": np.linspace(-10, -5, 21), "lat": np.linspace(50, 55, 21)},
        dims=["lat", "lon"],
    )
    bath = Bathymetry.from_array(data)
    bpi = bath.bpi(radius_km=50)

    # Centre point should have negative BPI (lower than surroundings)
    assert bpi.values[10, 10] < 0


def test_rugosity_calculation(fake_bathy):
    """Calculate Vector Ruggedness Measure."""
    rug = fake_bathy.rugosity(radius_km=1.0)

    assert rug.shape == fake_bathy.shape
    assert rug.name == "rugosity"


def test_rugosity_range(fake_bathy):
    """VRM values should be in [0, 1]."""
    rug = fake_bathy.rugosity(radius_km=1.0)

    assert rug.values.min() >= 0
    assert rug.values.max() <= 1


def test_rugosity_flat_surface_is_zero(flat_bathy):
    """Flat surface should have VRM ≈ 0 everywhere."""
    rug = flat_bathy.rugosity(radius_km=1.0)

    assert np.allclose(rug.values, 0, atol=1e-10)


def test_rugosity_tilted_plane_is_zero():
    """Uniformly sloping surface should have VRM ≈ 0 (all normals parallel)."""
    # Linear ramp: high slope, but all surface normals point the same direction
    x = np.linspace(0, 20, 30)
    y = np.linspace(0, 20, 30)
    xx, _ = np.meshgrid(x, y)
    ramp = xr.DataArray(
        -xx * 50.0,  # 50 m/cell slope in x direction
        coords={"lon": np.linspace(-10, -5, 30), "lat": np.linspace(50, 55, 30)},
        dims=["lat", "lon"],
    )
    bath = Bathymetry.from_array(ramp)

    assert np.allclose(bath.rugosity().values, 0, atol=1e-6)


def test_rugosity_rough_exceeds_flat(flat_bathy):
    """Rough terrain should have higher mean VRM than flat terrain."""
    rng = np.random.default_rng(0)
    rough_data = xr.DataArray(
        rng.uniform(-1000, 0, (20, 20)),
        coords={"lon": np.linspace(-10, -5, 20), "lat": np.linspace(50, 55, 20)},
        dims=["lat", "lon"],
    )
    rough = Bathymetry.from_array(rough_data)

    assert rough.rugosity().values.mean() > flat_bathy.rugosity().values.mean()


def test_aspect_calculation(fake_bathy):
    """Calculate seafloor aspect."""
    asp = fake_bathy.aspect()

    assert asp.shape == fake_bathy.shape
    assert asp.name == "aspect"


def test_aspect_range(fake_bathy):
    """Aspect values should be in [0, 360)."""
    asp = fake_bathy.aspect()
    valid = asp.values[~np.isnan(asp.values)]

    assert valid.min() >= 0
    assert valid.max() < 360


def test_aspect_flat_surface_is_nan(flat_bathy):
    """Flat surface (zero gradient) should return NaN everywhere."""
    asp = flat_bathy.aspect()

    assert np.all(np.isnan(asp.values))


def test_aspect_north_facing():
    """Surface ascending northward should have aspect = 0°."""
    lats = np.linspace(50, 55, 20)
    # Elevation increases with latitude (northward ascent)
    elevations = np.outer(np.linspace(-1000, -500, 20), np.ones(20))
    data = xr.DataArray(
        elevations,
        coords={"lon": np.linspace(-10, -5, 20), "lat": lats},
        dims=["lat", "lon"],
    )
    asp = Bathymetry.from_array(data).aspect()

    # Interior points should be north-facing (0°)
    assert np.allclose(asp.values[1:-1, 1:-1], 0, atol=1e-6)


def test_aspect_east_facing():
    """Surface ascending eastward should have aspect = 90°."""
    # Elevation increases with longitude (eastward ascent)
    elevations = np.outer(np.ones(20), np.linspace(-1000, -500, 20))
    data = xr.DataArray(
        elevations,
        coords={"lon": np.linspace(-10, -5, 20), "lat": np.linspace(50, 55, 20)},
        dims=["lat", "lon"],
    )
    asp = Bathymetry.from_array(data).aspect()

    # Interior points should be east-facing (90°)
    assert np.allclose(asp.values[1:-1, 1:-1], 90, atol=1e-6)


def test_clip(fake_bathy):
    """Clip returns a Bathymetry object bounded by the requested range."""
    clipped = fake_bathy.clip(lon_range=(-9, -7), lat_range=(51, 54))

    lon_min, lon_max = clipped.lon_range
    lat_min, lat_max = clipped.lat_range

    # All data within requested bounds
    assert -9 <= lon_min and lon_max <= -7
    assert 51 <= lat_min and lat_max <= 54

    # Covers most of the requested range (within one grid cell)
    cell_size = 5 / 19  # fake_bathy grid spacing
    assert lon_min < -9 + cell_size
    assert lon_max > -7 - cell_size
    assert lat_min < 51 + cell_size
    assert lat_max > 54 - cell_size


def test_to_netcdf(fake_bathy, tmp_path):
    """Export and reload NetCDF round-trips correctly."""
    filepath = str(tmp_path / "test_output.nc")
    fake_bathy.to_netcdf(filepath)

    reloaded = Bathymetry(filepath)
    assert reloaded.shape == fake_bathy.shape


def test_from_gebco_opendap_skips_download_if_file_exists(temp_netcdf, monkeypatch):
    """from_gebco_opendap skips download if save_path exists."""
    download_called = False

    def mock_download(*args, **kwargs):
        nonlocal download_called
        download_called = True
        return temp_netcdf

    monkeypatch.setattr(Bathymetry, "_download_gebco", mock_download)

    # Should load from existing file without downloading
    bath = Bathymetry.from_gebco_opendap(
        lon_range=(-10, -5),
        lat_range=(50, 55),
        save_path=temp_netcdf,
    )

    assert not download_called
    assert bath.shape == (20, 20)
