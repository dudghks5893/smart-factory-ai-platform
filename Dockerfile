# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12.14-slim-bookworm

FROM ghcr.io/astral-sh/uv:0.12.5 AS uv-bin

FROM ${PYTHON_IMAGE} AS runtime-dependencies
COPY --from=uv-bin /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-group dashboard --no-install-project

FROM ${PYTHON_IMAGE} AS dashboard-dependencies
COPY --from=uv-bin /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --only-group dashboard --no-install-project

FROM runtime-dependencies AS test-dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

FROM ${PYTHON_IMAGE} AS application-base
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache
WORKDIR /app
RUN groupadd --gid 10001 app && \
    useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

FROM application-base AS application
COPY alembic.ini ./
COPY configs ./configs
COPY migrations ./migrations
COPY ml ./ml
COPY pipelines ./pipelines
COPY services ./services
COPY shared ./shared

FROM application AS runtime
COPY --from=runtime-dependencies --chown=app:app /app/.venv /app/.venv
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"]
CMD ["uvicorn", "services.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM application AS test
COPY --from=test-dependencies --chown=app:app /app/.venv /app/.venv
COPY tests ./tests
USER app
CMD ["python", "-m", "pytest"]

FROM application-base AS dashboard-runtime
COPY --from=dashboard-dependencies --chown=app:app /app/.venv /app/.venv
COPY apps ./apps
COPY shared ./shared
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2).read()"]
CMD ["streamlit", "run", "apps/dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
