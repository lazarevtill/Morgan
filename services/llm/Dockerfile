# Multi-stage Dockerfile for LLM Service (OpenAI-compatible Ollama client)
# This service acts as a client to external Ollama, no CUDA needed

# Use shared base image
FROM morgan-base AS python-deps

WORKDIR /app

# Copy pyproject.toml and install LLM-specific dependencies
COPY pyproject.toml .
RUN uv pip install --system openai

# Application build
FROM python-deps AS build

WORKDIR /app

# Copy LLM service code
COPY services/llm/ ./

# Create necessary directories
RUN mkdir -p logs data/models

# Runtime image
FROM build AS runtime

WORKDIR /app

# Expose ports
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the LLM API service
CMD ["python", "main.py"]
