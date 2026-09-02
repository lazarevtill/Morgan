# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

FROM base AS runtime
COPY pyproject.toml ./
COPY morgan_brain ./morgan_brain
RUN pip install --no-cache-dir -e .
# The one database. docker-compose mounts ./data here; keep the path explicit rather than
# relying on the per-user default, which inside a container would be /root/.local/share.
ENV MORGAN_DATA_DIR=/app/data
EXPOSE 8080
# Default command is overridden per-service in docker-compose.
CMD ["python", "-m", "morgan_brain.apps.brain_api"]
