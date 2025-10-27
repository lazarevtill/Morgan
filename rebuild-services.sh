#!/bin/bash

# Morgan AI Services - Rebuild Script
# This script rebuilds all services with fixes applied

set -e

echo "================================================"
echo "Morgan AI Services - Rebuild Script"
echo "================================================"
echo ""

# Stop all services
echo "1. Stopping all services..."
docker-compose down

# Clean up old images (optional - uncomment if needed)
# echo "2. Removing old images..."
# docker rmi morgan-core:latest morgan-llm:latest morgan-tts:latest morgan-stt:latest || true

# Rebuild all services
echo "2. Rebuilding all services..."
docker-compose build --no-cache

# Create necessary directories for model caching
echo "3. Creating model cache directories..."
mkdir -p data/models/huggingface
mkdir -p data/models/transformers
mkdir -p data/models/torch_hub/hub
mkdir -p data/models/sentence_transformers
mkdir -p data/models/tts
mkdir -p data/models/stt
mkdir -p logs/core
mkdir -p logs/llm
mkdir -p logs/tts
mkdir -p logs/stt

# Set proper permissions
echo "4. Setting permissions..."
chmod -R 777 data/models
chmod -R 777 logs

# Start services
echo "5. Starting services..."
docker-compose up -d

# Wait for services to start
echo "6. Waiting for services to initialize..."
sleep 10

# Check service health
echo "7. Checking service health..."
echo ""
echo "Core Service:"
curl -s http://localhost:8000/health | jq '.' || echo "Core service not ready"
echo ""
echo "LLM Service:"
curl -s http://localhost:8001/health | jq '.' || echo "LLM service not ready"
echo ""
echo "TTS Service:"
curl -s http://localhost:8002/health | jq '.' || echo "TTS service not ready"
echo ""
echo "STT Service:"
curl -s http://localhost:8003/health | jq '.' || echo "STT service not ready"
echo ""

echo "================================================"
echo "Rebuild complete!"
echo "================================================"
echo ""
echo "View logs with: docker-compose logs -f [service_name]"
echo "Services:"
echo "  - core (http://localhost:8000)"
echo "  - llm-service (http://localhost:8001)"
echo "  - tts-service (http://localhost:8002)"
echo "  - stt-service (http://localhost:8003)"
echo ""

