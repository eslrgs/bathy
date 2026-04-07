"""Tests for bathy.grid module."""

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from shapely.geometry import box

from bathy.grid import clip, fill_gaps, merge, reproject, resample

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def geo_data():
    """Geographic DataArray with CRS set."""
    da = xr.DataArray(
        np.random.default_rng(42).random((20, 30)) * -100,
        coords={"lon": np.linspace(-10, -5, 30), "lat": np.linspace(50, 55, 20)},
        dims=["lat", "lon"],
        name="elevation",
    )
    da = da.rio.write_crs("EPSG:4326")
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    return da


@pytest.fixture
def projected_data():
    """Projected DataArray (UTM zone 29N)."""
    da = xr.DataArray(
        np.random.default_rng(42).random((20, 20)) * -100,
        coords={
            "x": np.linspace(500000, 510000, 20),
            "y": np.linspace(5500000, 5510000, 20),
        },
        dims=["y", "x"],
        name="elevation",
    )
    da = da.rio.write_crs("EPSG:32629")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return da


@pytest.fixture
def data_with_gaps(geo_data):
    """Geographic DataArray with NaN gaps."""
    arr = geo_data.values.copy()
    arr[5:10, 10:15] = np.nan
    return geo_data.copy(data=arr)


# ============================================================================
# clip
# ============================================================================


class TestClip:
    def test_clip_with_geodataframe(self, geo_data):
        gdf = gpd.GeoDataFrame(geometry=[box(-8, 51, -7, 53)], crs="EPSG:4326")
        result = clip(geo_data, geometry=gdf)
        assert result.lon.min() >= -8
        assert result.lon.max() <= -7
        assert result.lat.min() >= 51
        assert result.lat.max() <= 53

    def test_clip_with_region(self):
        """Use data that overlaps the north_sea region (-4..9, 51..62)."""
        da = xr.DataArray(
            np.random.default_rng(42).random((20, 20)) * -100,
            coords={"lon": np.linspace(-5, 10, 20), "lat": np.linspace(49, 63, 20)},
            dims=["lat", "lon"],
            name="elevation",
        )
        da = da.rio.write_crs("EPSG:4326")
        da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        result = clip(da, region="north_sea")
        assert result.sizes["lon"] > 0
        assert result.sizes["lat"] > 0

    def test_clip_dims_geographic(self, geo_data):
        gdf = gpd.GeoDataFrame(geometry=[box(-8, 51, -7, 53)], crs="EPSG:4326")
        result = clip(geo_data, geometry=gdf)
        assert "lon" in result.dims
        assert "lat" in result.dims

    def test_clip_requires_exactly_one_arg(self, geo_data):
        with pytest.raises(ValueError, match="exactly one"):
            clip(geo_data)
        with pytest.raises(ValueError, match="exactly one"):
            clip(geo_data, geometry=gpd.GeoDataFrame(), region="north_sea")

    def test_clip_invalid_region(self, geo_data):
        with pytest.raises(ValueError, match="Unknown region"):
            clip(geo_data, region="atlantis")


# ============================================================================
# resample
# ============================================================================


class TestResample:
    def test_resample_coarser(self, geo_data):
        original_res = float(np.abs(np.diff(geo_data.lon.values).mean()))
        coarser_res = original_res * 2
        result = resample(geo_data, resolution_degrees=coarser_res)
        assert result.sizes["lon"] < geo_data.sizes["lon"]

    def test_resample_finer(self, geo_data):
        original_res = float(np.abs(np.diff(geo_data.lon.values).mean()))
        finer_res = original_res / 2
        result = resample(geo_data, resolution_degrees=finer_res)
        assert result.sizes["lon"] > geo_data.sizes["lon"]

    def test_resample_preserves_crs(self, geo_data):
        result = resample(geo_data, resolution_degrees=0.5)
        assert result.rio.crs is not None

    def test_resample_dims_normalised(self, geo_data):
        result = resample(geo_data, resolution_degrees=0.5)
        assert "lon" in result.dims or "x" in result.dims

    def test_resample_invalid_method(self, geo_data):
        with pytest.raises(ValueError, match="Unknown method"):
            resample(geo_data, resolution_degrees=0.5, method="quadratic")

    def test_resample_projected(self, projected_data):
        result = resample(projected_data, resolution_m=1000)
        assert "x" in result.dims
        assert "y" in result.dims

    def test_resample_resolution_m(self, geo_data):
        result = resample(geo_data, resolution_m=50_000)
        assert result.sizes["lon"] < geo_data.sizes["lon"]
        assert result.sizes["lat"] < geo_data.sizes["lat"]

    def test_resample_requires_exactly_one(self, geo_data):
        with pytest.raises(ValueError, match="exactly one"):
            resample(geo_data)
        with pytest.raises(ValueError, match="exactly one"):
            resample(geo_data, resolution_degrees=0.5, resolution_m=500)


# ============================================================================
# reproject
# ============================================================================


class TestReproject:
    def test_reproject_to_utm(self, geo_data):
        result = reproject(geo_data, target_crs="EPSG:32629")
        assert "x" in result.dims
        assert "y" in result.dims
        assert result.rio.crs.to_epsg() == 32629

    def test_reproject_to_geographic(self, projected_data):
        result = reproject(projected_data, target_crs="EPSG:4326")
        assert "lon" in result.dims
        assert "lat" in result.dims

    def test_reproject_with_resolution(self, geo_data):
        result = reproject(geo_data, target_crs="EPSG:32629", resolution=5000)
        assert result.rio.crs.to_epsg() == 32629

    def test_reproject_invalid_method(self, geo_data):
        with pytest.raises(ValueError, match="Unknown method"):
            reproject(geo_data, target_crs="EPSG:32629", method="bad")


# ============================================================================
# merge
# ============================================================================


class TestMerge:
    def _make_tiles(self):
        """Create two overlapping geographic tiles."""
        tile_a = xr.DataArray(
            np.full((10, 10), -100.0),
            coords={"lon": np.linspace(-10, -6, 10), "lat": np.linspace(50, 54, 10)},
            dims=["lat", "lon"],
            name="elevation",
        )
        tile_a = tile_a.rio.write_crs("EPSG:4326")
        tile_a = tile_a.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

        tile_b = xr.DataArray(
            np.full((10, 10), -200.0),
            coords={"lon": np.linspace(-8, -4, 10), "lat": np.linspace(50, 54, 10)},
            dims=["lat", "lon"],
            name="elevation",
        )
        tile_b = tile_b.rio.write_crs("EPSG:4326")
        tile_b = tile_b.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        return tile_a, tile_b

    def test_merge_mean(self):
        a, b = self._make_tiles()
        result = merge([a, b], method="mean")
        assert result.lon.min() <= -10
        assert result.lon.max() >= -4

    def test_merge_min(self):
        a, b = self._make_tiles()
        result = merge([a, b], method="min")
        # In overlap, min of -100 and -200 is -200
        overlap = result.sel(lon=slice(-8, -6), lat=slice(50, 54))
        assert float(overlap.min()) <= -100

    def test_merge_max(self):
        a, b = self._make_tiles()
        result = merge([a, b], method="max")
        overlap = result.sel(lon=slice(-8, -6), lat=slice(50, 54))
        assert float(overlap.max()) >= -200

    def test_merge_first(self):
        a, b = self._make_tiles()
        result = merge([a, b], method="first")
        assert result.sizes["lon"] > 0

    def test_merge_needs_two(self, geo_data):
        with pytest.raises(ValueError, match="at least 2"):
            merge([geo_data])

    def test_merge_invalid_method(self):
        a, b = self._make_tiles()
        with pytest.raises(ValueError, match="Unknown method"):
            merge([a, b], method="sum")


# ============================================================================
# fill_gaps
# ============================================================================


class TestFillGaps:
    def test_fill_nearest(self, data_with_gaps):
        result = fill_gaps(data_with_gaps, method="nearest")
        assert not np.isnan(result.values).any()

    def test_fill_linear(self, data_with_gaps):
        result = fill_gaps(data_with_gaps, method="linear")
        # Linear may leave NaN at edges, but interior should be filled
        interior = result.values[6:9, 11:14]
        assert not np.isnan(interior).any()

    def test_fill_no_gaps(self, geo_data):
        result = fill_gaps(geo_data)
        xr.testing.assert_equal(result, geo_data)

    def test_fill_invalid_method(self, data_with_gaps):
        with pytest.raises(ValueError, match="Unknown method"):
            fill_gaps(data_with_gaps, method="cubic")
