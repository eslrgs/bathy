"""Tests for io module."""

import pytest

import bathy.io as io_module
from bathy.io import (
    list_regions,
    load_bathymetry,
    load_emodnet_wcs,
    load_etopo,
    load_gebco_opendap,
    load_noaa_crm,
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


# === EMODnet tests ===


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "Must specify"),
        ({"region": "north_sea", "lon_range": (-4, 9)}, "Cannot specify both"),
    ],
)
def test_emodnet_rejects_invalid_args(kwargs, match):
    """load_emodnet_wcs validates region/bounds arguments."""
    with pytest.raises(ValueError, match=match):
        load_emodnet_wcs(**kwargs)


def test_emodnet_caching_and_download(temp_geotiff, tmp_path, monkeypatch):
    """load_emodnet_wcs skips download when cached, downloads otherwise."""
    import shutil

    calls = []
    call_counter = [0]

    def mock_download(lon_range, lat_range, save_path):
        calls.append({"lon_range": lon_range, "lat_range": lat_range})
        # Return a fresh copy each time so cleanup doesn't affect other calls
        copy = tmp_path / f"emodnet_{call_counter[0]}.tif"
        call_counter[0] += 1
        shutil.copy(temp_geotiff, copy)
        return str(copy)

    monkeypatch.setattr(io_module, "_download_emodnet", mock_download)

    # Cached file — no download
    data = load_emodnet_wcs(
        lon_range=(-10, -5), lat_range=(50, 55), save_path=temp_geotiff
    )
    assert len(calls) == 0
    assert "lon" in data.dims or "x" in data.dims

    # No cache — triggers download
    load_emodnet_wcs(lon_range=(-10, -5), lat_range=(50, 55))
    assert len(calls) == 1

    # Region preset — resolves bounds and downloads
    load_emodnet_wcs(region="north_sea")
    assert calls[-1]["lon_range"] == (-4, 9)
    assert calls[-1]["lat_range"] == (51, 62)


# === ETOPO tests ===


def test_etopo_rejects_invalid_resolution():
    """load_etopo rejects invalid resolution strings."""
    with pytest.raises(ValueError, match="Invalid resolution"):
        load_etopo(lon_range=(-10, -5), lat_range=(50, 55), resolution="10s")


def test_etopo_rejects_missing_bounds():
    """load_etopo requires bounds or region."""
    with pytest.raises(ValueError, match="Must specify"):
        load_etopo()


def test_etopo_loads_from_cache(temp_netcdf_z):
    """load_etopo loads from save_path without network access."""
    data = load_etopo(
        lon_range=(-10, -5),
        lat_range=(50, 55),
        save_path=temp_netcdf_z,
    )
    assert data.shape == (20, 20)


# === NOAA CRM tests ===


def test_noaa_crm_rejects_missing_bounds():
    """load_noaa_crm requires bounds or region."""
    with pytest.raises(ValueError, match="Must specify"):
        load_noaa_crm()


def test_noaa_crm_rejects_non_us_region():
    """load_noaa_crm rejects regions outside US waters."""
    with pytest.raises(ValueError, match="does not overlap"):
        load_noaa_crm(lon_range=(0, 10), lat_range=(50, 55))


def test_find_crm_volume():
    """_find_crm_volume selects correct volume for known regions."""
    from bathy.io import _find_crm_volume

    # US East Coast (vol1: -80 to -64, 40 to 48)
    assert _find_crm_volume((-72, -70), (41, 43)) == 1
    # Hawaii (vol10: -162 to -152, 18 to 24)
    assert _find_crm_volume((-158, -155), (19, 21)) == 10
    # Gulf of Mexico (vol4 or 5)
    assert _find_crm_volume((-96, -95), (26, 28)) in (4, 5)
