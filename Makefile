.PHONY: format test

format:
	@echo "Running ruff..."
	uv run ruff format .
	uv run ruff check . --fix

test:
	@echo "Running tests..."
	uv run pytest