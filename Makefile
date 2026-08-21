.PHONY: sync format format-check lint typecheck test check docker-build docker-up docker-down docker-clean-volumes docker-test monitoring-up monitoring-down monitoring-config-check k8s-render k8s-check

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

monitoring-up:
	docker compose up --detach prometheus grafana

monitoring-down:
	docker compose stop prometheus grafana

monitoring-config-check:
	docker run --rm --entrypoint /bin/promtool --volume "$(CURDIR)/monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro" prom/prometheus:v3.12.0 check config /etc/prometheus/prometheus.yml

k8s-render:
	kubectl kustomize infra/k8s/base
	kubectl kustomize infra/k8s/overlays/local-cpu
	kubectl kustomize infra/k8s/overlays/gcp-gpu

k8s-check:
	kubectl kustomize infra/k8s/base > /dev/null
	kubectl kustomize infra/k8s/overlays/local-cpu > /dev/null
	kubectl kustomize infra/k8s/overlays/gcp-gpu > /dev/null
