# Installation

## Requirements

- Python 3.12 or later
- NumPy, xarray, matplotlib, scipy, and other scientific Python packages

## Install from source

Clone the repository and install with uv:

```bash
git clone https://github.com/eslrgs/bathy.git
cd bathy
uv sync
```

Or with pip:

```bash
git clone https://github.com/eslrgs/bathy.git
cd bathy
pip install -e .
```

## Optional dependencies

For Jupyter notebook support:

```bash
uv sync --extra notebook
```
