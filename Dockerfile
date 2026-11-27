# syntax=docker/dockerfile:1
# Keep Python's supported 3.12 line and Debian release explicit. Refresh this multi-platform
# digest with reviewed security updates; it is an artifact pin, not a promise of no CVEs.
FROM python:3.12.16-slim-trixie@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# Security fixes may reach Debian after the pinned image was published. Apply available
# updates in this layer; rebuild without cache to refresh them. The image digest pins the OS.
RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY morgan_brain ./morgan_brain
# ensurepip in the supported Python image still supplies pip 25.0.1. Update the installer
# before resolving/building Morgan, and verify the resulting dependency graph.
RUN python -m pip install --no-cache-dir --upgrade "pip>=26.3,<27" \
    && python -m pip install --no-cache-dir . \
    && python -m pip check
# The one database. docker-compose mounts ./data here; keep the path explicit rather than
# relying on the per-user default, which inside a container would be /root/.local/share.
ENV MORGAN_DATA_DIR=/app/data
EXPOSE 8090
# The MCP server over streamable-HTTP. A container's published port is unreachable from a
# loopback bind, so this binds all interfaces -- which means MORGAN_API_KEY must be a real
# key or the server refuses to start. That refusal is the point.
CMD ["morgan-mcp", "--transport", "http", "--host", "0.0.0.0"]
