"""Tests for io module."""

import bathy.io as io_module
from bathy.io import (
    list_regions,
    load_bathymetry,
    load_gebco_opendap,
)


def test_list_regions():
    """List all preset regions."""
    regions = list_regions()

    assert "mediterranean" in regions
    assert "mariana_trench" in regions


def test_load_from_netcdf(temp_netcdf):
    """Load bathymetry from NetCDF file."""
    data = load_bathymetry(temp_netcdf)

    assert data.shape == (20, 20)
    assert (float(data.lon.min()), float(data.lon.max())) == (-10.0, -5.0)
    assert (float(data.lat.min()), float(data.lat.max())) == (50.0, 55.0)


def test_to_netcdf(fake_bathy, tmp_path):
    """Export and reload NetCDF round-trips correctly."""
    filepath = str(tmp_path / "test_output.nc")
    fake_bathy.to_netcdf(filepath)

    reloaded = load_bathymetry(filepath)
    assert reloaded.shape == fake_bathy.shape


def test_from_gebco_opendap_skips_download_if_file_exists(temp_netcdf, monkeypatch):
    """load_gebco_opendap skips download if save_path exists."""
    download_called = False

    def mock_download(*args, **kwargs):
        nonlocal download_called
        download_called = True
        return temp_netcdf

    monkeypatch.setattr(io_module, "_download_gebco", mock_download)

    data = load_gebco_opendap(
        lon_range=(-10, -5),
        lat_range=(50, 55),
        save_path=temp_netcdf,
    )

    assert not download_called
    assert data.shape == (20, 20)
