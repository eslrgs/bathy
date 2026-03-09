"""Launch the interactive profile drawer from the command line.

Usage
-----
    uv run bathy-draw path/to/data.nc
    uv run python -m bathy path/to/data.nc
"""

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: bathy-draw <path/to/data.nc>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    from bathy.draw import draw_profile
    from bathy.io import load_bathymetry
    from bathy.profile import to_gdf

    data = load_bathymetry(str(path))
    profiles = draw_profile(data)

    if not profiles:
        print("No profiles drawn.")
        return

    out_path = path.with_name(f"{path.stem}_profiles.gpkg")
    to_gdf(profiles).to_file(out_path)

    print(f"\n{len(profiles)} profile(s) saved to {out_path}")
    for p in profiles:
        print(
            f"  {p.name}: {p.distances[-1] / 1000:.1f} km, "
            f"min elevation {float(p.elevations.min()):.0f} m"
        )


if __name__ == "__main__":
    main()
