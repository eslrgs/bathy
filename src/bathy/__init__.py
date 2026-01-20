"""Lightweight Python package for exploring bathymetry data."""

import logging

from bathy import profile
from bathy.bathymetry import Bathymetry, list_regions
from bathy.profile import Profile

__all__ = ["Bathymetry", "Profile", "profile", "list_regions"]

# Set up default logging configuration
logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(name)s: %(message)s")
