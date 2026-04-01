# CLAUDE.md

Lightweight Python package for exploring bathymetric data, targeting academic users.

## Project structure

- `src/bathy/` — package source
- `tests/` — pytest test suite
- `docs/` — MkDocs documentation source
- `examples/` — example `.ipynb` notebooks and `.py` scripts

## Tooling

- **uv** for package management and running commands
- **just** as the task runner
- **ruff** for formatting and linting
- **ty** for type checking

## Code style

- Keep code simple and minimal — no over-engineering
- NumPy-style docstrings
- Python 3.12+, use ty-compatible types (e.g. `tuple[float, float] | None`)
- Grid methods return `xarray.DataArray`; tabular methods return `polars.DataFrame`
- Don't use OOP architecture — use functions, not class hierarchies; dataclasses for data containers are fine

## Conventions

- Depths are **negative** (below sea level); positive = land
- Coordinate tuples use **(lon, lat)** order, e.g. `start=(lon, lat)`
- xarray dims: `"lon"` / `"lat"` for geographic CRS, `"x"` / `"y"` for projected CRS — use `get_xy_dims()` to detect
- Units: elevation in metres, distance in metres, slope in degrees
- Propagate NaNs honestly — do not silently fill or mask them
