"""Standalone smoke test for CI publish pipeline. No pytest required."""

import bathy

assert bathy.__version__, "version is missing"

# Check key public functions are importable
for name in [
    "load_bathymetry",
    "slope",
    "plot_bathy",
    "extract_profile",
    "get_canyons",
]:
    assert hasattr(bathy, name), f"missing public function: {name}"

print(f"bathy {bathy.__version__} OK")
