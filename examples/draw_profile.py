"""Profile drawing with PyQt6.

Requires: uv pip install bathy[draw]

Usage: uv run python examples/draw_profile.py
   or: uv run bathy-draw data/ne_atlantic_gebco.nc
"""

import bathy

data = bathy.load_bathymetry("data/ne_atlantic_gebco.nc")
profiles = bathy.draw_profile(data)

print(f"\n{len(profiles)} profile(s) extracted")
for p in profiles:
    print(
        f"  {p.name}: {p.distances[-1] / 1000:.1f} km, "
        f"min elevation {float(p.elevations.min()):.0f} m"
    )
