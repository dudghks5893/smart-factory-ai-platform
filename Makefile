.PHONY: sync format format-check lint typecheck test check docker-build docker-up docker-down docker-clean-volumes docker-test

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

docker-build:
	docker compose build api

docker-up:
	docker compose up --detach --build api

docker-down:
	docker compose down --remove-orphans

docker-clean-volumes:
	docker compose down --volumes --remove-orphans

docker-test:
	./scripts/docker_test.sh
