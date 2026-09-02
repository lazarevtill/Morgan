# syntax=docker/dockerfile:1
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml ./
COPY morgan_brain ./morgan_brain
RUN pip install --no-cache-dir .
# The one database. docker-compose mounts ./data here; keep the path explicit rather than
# relying on the per-user default, which inside a container would be /root/.local/share.
ENV MORGAN_DATA_DIR=/app/data
EXPOSE 8090
# The MCP server over streamable-HTTP. A container's published port is unreachable from a
# loopback bind, so this binds all interfaces -- which means MORGAN_API_KEY must be a real
# key or the server refuses to start. That refusal is the point.
CMD ["morgan-mcp", "--transport", "http", "--host", "0.0.0.0"]
