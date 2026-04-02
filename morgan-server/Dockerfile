# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY morgan-rag/pyproject.toml morgan-rag/README.md /build/morgan-rag/
COPY morgan-rag/morgan /build/morgan-rag/morgan

COPY morgan-server/pyproject.toml morgan-server/README.md /build/morgan-server/
COPY morgan-server/morgan_server /build/morgan-server/morgan_server

RUN pip install --upgrade pip setuptools wheel && \
    pip install "/build/morgan-rag[server,channels,scheduling,tokens]" && \
    pip install /build/morgan-server

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 --shell /bin/bash morgan && \
    mkdir -p /app/data /app/config /home/morgan/.morgan && \
    chown -R morgan:morgan /app /home/morgan

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    MORGAN_HOST=0.0.0.0 \
    MORGAN_PORT=8080

USER morgan

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=5 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["python", "-m", "morgan_server"]
