#!/bin/bash

# Morgan AI Assistant - Service Startup Script
# This script builds and starts all Morgan services

set -e

echo "=== Morgan AI Assistant - Starting Services ==="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Function to print messages
print_status() {
    echo -e "${BLUE}>>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check if Docker is running
print_status "Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi
print_success "Docker is running"

# Check if docker-compose is available
print_status "Checking docker-compose..."
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose not found. Please install docker-compose."
    exit 1
fi
print_success "docker-compose is available"

# Stop existing services
print_status "Stopping existing services..."
docker-compose down > /dev/null 2>&1 || true
print_success "Existing services stopped"

# Check if images exist
print_status "Checking for local images..."
CORE_EXISTS=$(docker images -q morgan-core:latest)
LLM_EXISTS=$(docker images -q morgan-llm:latest)
TTS_EXISTS=$(docker images -q morgan-tts:latest)
STT_EXISTS=$(docker images -q morgan-stt:latest)

BUILD_NEEDED=false
if [ -z "$CORE_EXISTS" ]; then
    print_warning "Core image not found"
    BUILD_NEEDED=true
fi
if [ -z "$LLM_EXISTS" ]; then
    print_warning "LLM image not found"
    BUILD_NEEDED=true
fi
if [ -z "$TTS_EXISTS" ]; then
    print_warning "TTS image not found"
    BUILD_NEEDED=true
fi
if [ -z "$STT_EXISTS" ]; then
    print_warning "STT image not found"
    BUILD_NEEDED=true
fi

# Build images if needed
if [ "$BUILD_NEEDED" = true ] || [ "$1" = "--build" ] || [ "$1" = "-b" ]; then
    print_status "Building service images..."

    print_status "Building Core service..."
    docker build -t morgan-core:latest -f core/Dockerfile . > /tmp/build-core.log 2>&1
    if [ $? -eq 0 ]; then
        print_success "Core service built"
    else
        print_error "Core service build failed. Check /tmp/build-core.log"
        exit 1
    fi

    print_status "Building LLM service..."
    docker build -t morgan-llm:latest -f services/llm/Dockerfile . > /tmp/build-llm.log 2>&1
    if [ $? -eq 0 ]; then
        print_success "LLM service built"
    else
        print_error "LLM service build failed. Check /tmp/build-llm.log"
        exit 1
    fi

    print_status "Building TTS service (this may take a while)..."
    docker build -t morgan-tts:latest -f services/tts/Dockerfile . > /tmp/build-tts.log 2>&1 &
    TTS_PID=$!

    print_status "Building STT service (this may take a while)..."
    docker build -t morgan-stt:latest -f services/stt/Dockerfile . > /tmp/build-stt.log 2>&1 &
    STT_PID=$!

    # Wait for both builds
    wait $TTS_PID
    if [ $? -eq 0 ]; then
        print_success "TTS service built"
    else
        print_error "TTS service build failed. Check /tmp/build-tts.log"
        exit 1
    fi

    wait $STT_PID
    if [ $? -eq 0 ]; then
        print_success "STT service built"
    else
        print_error "STT service build failed. Check /tmp/build-stt.log"
        exit 1
    fi
else
    print_success "All images found, skipping build"
fi

# Create necessary directories
print_status "Creating directories..."
mkdir -p logs/core logs/llm logs/tts logs/stt
mkdir -p data/conversations data/models/llm data/models/tts data/models/stt data/voices
print_success "Directories created"

# Start services with docker-compose
print_status "Starting services with docker-compose..."
docker-compose up -d

# Wait for services to be healthy
print_status "Waiting for services to be healthy..."
sleep 5

# Check service health
RETRY=0
MAX_RETRIES=30

while [ $RETRY -lt $MAX_RETRIES ]; do
    CORE_HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

    if [ "$CORE_HEALTH" = "healthy" ] || [ "$CORE_HEALTH" = "degraded" ]; then
        print_success "Core service is $CORE_HEALTH"
        break
    fi

    RETRY=$((RETRY + 1))
    echo -n "."
    sleep 2
done

echo ""

if [ $RETRY -eq $MAX_RETRIES ]; then
    print_warning "Core service health check timed out"
else
    # Check individual services
    print_status "Checking individual services..."

    LLM_HEALTH=$(curl -s http://localhost:8001/health 2>/dev/null)
    if [ -n "$LLM_HEALTH" ]; then
        print_success "LLM service is responding"
    else
        print_warning "LLM service not responding"
    fi

    TTS_HEALTH=$(curl -s http://localhost:8002/health 2>/dev/null)
    if [ -n "$TTS_HEALTH" ]; then
        print_success "TTS service is responding"
    else
        print_warning "TTS service not responding"
    fi

    STT_HEALTH=$(curl -s http://localhost:8003/health 2>/dev/null)
    if [ -n "$STT_HEALTH" ]; then
        print_success "STT service is responding"
    else
        print_warning "STT service not responding"
    fi
fi

echo ""
echo "=== Morgan AI Assistant - Services Started ==="
echo ""
echo -e "${BLUE}Service URLs:${NC}"
echo -e "  Core:   ${GREEN}http://localhost:8000${NC}"
echo -e "  LLM:    ${GREEN}http://localhost:8001${NC}"
echo -e "  TTS:    ${GREEN}http://localhost:8002${NC}"
echo -e "  STT:    ${GREEN}http://localhost:8003${NC}"
echo ""
echo -e "${BLUE}API Documentation:${NC}"
echo -e "  Core:   ${GREEN}http://localhost:8000/docs${NC}"
echo -e "  LLM:    ${GREEN}http://localhost:8001/docs${NC}"
echo ""
echo -e "${BLUE}Health Check:${NC}"
echo -e "  ${GREEN}curl http://localhost:8000/health${NC}"
echo ""
echo -e "${BLUE}View Logs:${NC}"
echo -e "  ${GREEN}docker-compose logs -f core${NC}"
echo -e "  ${GREEN}docker-compose logs -f llm-service${NC}"
echo -e "  ${GREEN}docker-compose logs -f tts-service${NC}"
echo -e "  ${GREEN}docker-compose logs -f stt-service${NC}"
echo ""
echo -e "${BLUE}Stop Services:${NC}"
echo -e "  ${GREEN}docker-compose down${NC}"
echo ""
