# Installation

## Requirements

- Python 3.12 or later

## Install from PyPI

```bash
pip install bathy
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv add bathy        # add to a uv project
```

## Optional dependencies

For the interactive profile drawing tool (requires PyQt6):

```bash
pip install bathy[draw]
# or
uv add bathy[draw]
```

For Jupyter notebook support:

```bash
pip install bathy[notebook]
# or
uv add bathy[notebook]
```

## Install from source

```bash
git clone https://github.com/eslrgs/bathy.git
cd bathy
uv sync
```
