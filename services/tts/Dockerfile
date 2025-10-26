# Multi-stage Dockerfile for TTS Service with CUDA 13 support

# Use shared CUDA base image
FROM morgan-cuda-base AS python-deps

WORKDIR /app

# Copy pyproject.toml and install TTS-specific dependencies
COPY pyproject.toml .
RUN uv pip install --system \
    TTS \
    soundfile \
    librosa \
    pydub \
    torchvision \
    kokoro \
    pyttsx3

# Application build
FROM python-deps AS build

WORKDIR /app

# Copy TTS service code
COPY services/tts/ ./

# Create necessary directories
RUN mkdir -p logs data/models data/voices

# Runtime image
FROM build AS runtime

WORKDIR /app

# Expose ports
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the application
CMD ["python", "main.py"]
