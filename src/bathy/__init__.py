"""Lightweight Python package for exploring bathymetry data."""

import logging
from importlib.metadata import version

from bathy import profile
from bathy.analysis import (
    aspect,
    bpi,
    contours,
    curvature,
    geomorphons,
    hypsometric_curve,
    hypsometric_index,
    rugosity,
    slope,
    smooth,
    summary,
)
from bathy.datasets import sample_data
from bathy.grid import (
    clip,
    fill_gaps,
    merge,
    reproject,
    resample,
)
from bathy.io import (
    list_regions,
    load_bathymetry,
    load_emodnet_wcs,
    load_gebco_opendap,
    to_geotiff,
)
from bathy.plot import (
    plot_aspect,
    plot_bathy,
    plot_bpi,
    plot_curvature,
    plot_depth_zones,
    plot_geomorphons,
    plot_hillshade,
    plot_histogram,
    plot_hypsometric_curve,
    plot_interactive,
    plot_overview,
    plot_rugosity,
    plot_slope,
    plot_surface3d,
)
from bathy.profile import (
    Profile,
    compare_stats,
    concavity_index,
    cross_sections,
    extract_profile,
    get_canyons,
    gradient,
    knickpoints,
    max_depth,
    profile_from_coordinates,
    profile_stats,
    profiles_from_file,
    profiles_from_gdf,
    to_gdf,
)
from bathy.profile_plot import (
    plot_canyons,
    plot_gradient,
    plot_knickpoints,
    plot_profile,
    plot_profiles,
    plot_profiles_grid,
    plot_profiles_map,
)

__version__ = version("bathy")


def draw_profile(*args, **kwargs):
    """Lazy wrapper — see :func:`bathy.draw.draw_profile`."""
    from bathy.draw import draw_profile as _draw_profile

    return _draw_profile(*args, **kwargs)


__all__ = [
    # Datasets
    "sample_data",
    # Grid operations
    "clip",
    "resample",
    "reproject",
    "merge",
    "fill_gaps",
    # IO
    "list_regions",
    "load_bathymetry",
    "load_emodnet_wcs",
    "load_gebco_opendap",
    "to_geotiff",
    # Analysis
    "summary",
    "hypsometric_index",
    "hypsometric_curve",
    "slope",
    "curvature",
    "bpi",
    "rugosity",
    "aspect",
    "geomorphons",
    "contours",
    "smooth",
    # Plotting
    "plot_bathy",
    "plot_hillshade",
    "plot_slope",
    "plot_curvature",
    "plot_bpi",
    "plot_rugosity",
    "plot_aspect",
    "plot_geomorphons",
    "plot_overview",
    "plot_histogram",
    "plot_depth_zones",
    "plot_surface3d",
    "plot_hypsometric_curve",
    # Profile
    "Profile",
    "extract_profile",
    "profile_from_coordinates",
    "cross_sections",
    "profiles_from_file",
    "profiles_from_gdf",
    "profile_stats",
    "max_depth",
    "gradient",
    "concavity_index",
    "knickpoints",
    "get_canyons",
    "compare_stats",
    "to_gdf",
    "plot_profile",
    "plot_profiles",
    "plot_profiles_grid",
    "plot_profiles_map",
    "plot_knickpoints",
    "plot_gradient",
    "plot_canyons",
    # Interactive
    "plot_interactive",
    "draw_profile",
    # Submodule
    "profile",
    "__version__",
]

logging.getLogger("bathy").addHandler(logging.NullHandler())
