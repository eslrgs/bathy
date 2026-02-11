"""Lightweight Python package for exploring bathymetry data."""

import logging
from importlib.metadata import version

from bathy import profile
from bathy.bathymetry import Bathymetry, list_regions
from bathy.profile import Profile

__version__ = version("bathy")
__all__ = ["Bathymetry", "Profile", "profile", "list_regions", "__version__"]

# Set up default logging configuration
logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(name)s: %(message)s")
