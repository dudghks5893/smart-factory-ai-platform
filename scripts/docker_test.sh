#!/bin/sh
set -eu

project_name="${COMPOSE_PROJECT_NAME:-smartfactory-step7-test}"
if [ "${project_name}" = "smartfactory" ]; then
    echo "Refusing to use the persistent development Compose project for ephemeral tests." >&2
    exit 2
fi

cleanup() {
    docker compose --project-name "${project_name}" --profile test down --volumes --remove-orphans
}

trap cleanup EXIT INT TERM

docker compose \
    --project-name "${project_name}" \
    --profile test \
    up --build --abort-on-container-exit --exit-code-from test test
