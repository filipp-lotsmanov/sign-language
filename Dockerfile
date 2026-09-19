# Serving image. Model weights are NOT baked in: they are large, they are
# release artifacts, and rebuilding the image to change a checkpoint is wasteful.
# Mount them at /app/models, or fetch them in an entrypoint.
FROM python:3.12-slim AS base

# OpenCV and MediaPipe link against these; the slim image does not carry them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Dependencies first, so a source change does not invalidate the install layer.
# --frozen fails if uv.lock is stale rather than silently resolving something else.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# README.md is required: pyproject declares it via `readme`, so the project
# install in the next step fails without it.
COPY README.md main.py ./
COPY src/ ./src/
COPY frontend/ ./frontend/

RUN uv sync --frozen --no-dev

# Inside a container the app must listen on all interfaces; publish the port
# only where you intend it to be reachable. The WebSocket endpoint is
# unauthenticated, so put an authenticating proxy in front of it if that is not
# a trusted network. See SECURITY.md.
ENV HOST=0.0.0.0 \
    PORT=8000

# Drop privileges.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Reports unhealthy when hand detection failed to initialise, which is what
# happens when models/hand_landmarker.task was not mounted.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["python", "main.py"]
