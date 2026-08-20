.PHONY: sync format format-check lint typecheck test check

sync:
	uv sync

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

typecheck:
	uv run mypy ml pipelines services shared tests migrations

test:
	uv run python -m pytest

check: format-check lint typecheck test
