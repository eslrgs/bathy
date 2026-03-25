# Format and lint
format:
    uv run ruff format .
    uv run ruff check . --fix

# Run tests
test:
    uv run pytest

# Serve docs locally
docs:
    uv run mkdocs serve

# Build docs
docs-build:
    uv run mkdocs build

# Build package 
build:
    uv build

# Show README
readme:
    uvx grip README.md

# Type check
typecheck:
    uvx ty check .