#!/bin/bash
set -e

echo "Building Core service..."
docker build -t morgan-core:latest -f core/Dockerfile . 2>&1 | tail -5

echo "Building LLM service..."
docker build -t morgan-llm:latest -f services/llm/Dockerfile . 2>&1 | tail -5

echo "Building TTS service..."
docker build -t morgan-tts:latest -f services/tts/Dockerfile . 2>&1 | tail -5

echo "Building STT service..."
docker build -t morgan-stt:latest -f services/stt/Dockerfile . 2>&1 | tail -5

echo "All services built successfully!"
docker images | grep morgan
