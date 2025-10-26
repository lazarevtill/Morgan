# Multi-stage Dockerfile for STT Service with CUDA 13 support

# Use shared CUDA base image
FROM morgan-cuda-base AS python-deps

WORKDIR /app

# Copy pyproject.toml and install STT-specific dependencies
COPY pyproject.toml .
RUN uv pip install --system \
    faster-whisper \
    soundfile \
    librosa \
    pydub \
    silero-vad

# Application build
FROM python-deps AS build

WORKDIR /app

# Copy shared utilities first
COPY shared/ ./shared/

# Copy STT service code
COPY services/stt/ ./

# Create necessary directories
RUN mkdir -p logs data/models

# Runtime image
FROM build AS runtime

WORKDIR /app

# Set environment variables
ENV RUST_BACKTRACE=1

# Expose ports
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Run the application
CMD ["python", "main.py"]
