.PHONY: setup check test smoke clean

setup:
	uv sync --locked

check:
	uv run --locked ruff check .
	uv run --locked ruff format --check .
	uv run --locked pyright

test:
	uv run --locked python -m pytest

# Makes one billable UNC Azure Responses API request.
smoke:
	uv run --locked --env-file .env openai-smoke

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov build dist
	rm -f .coverage .coverage.*
	find src tests -type d \( -name __pycache__ -o -name '*.egg-info' \) -prune -exec rm -rf {} +
