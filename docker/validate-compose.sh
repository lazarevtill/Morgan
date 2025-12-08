#!/bin/bash
# Validation script for Docker Compose configuration

set -e

echo "=== Morgan Docker Compose Validation ==="
echo ""

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed"
    exit 1
fi
echo "✓ docker-compose is installed"

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running"
    exit 1
fi
echo "✓ Docker daemon is running"

# Validate docker-compose.yml syntax
echo ""
echo "Validating docker-compose.yml..."
if docker-compose -f docker-compose.yml config --quiet; then
    echo "✓ docker-compose.yml is valid"
else
    echo "❌ docker-compose.yml has syntax errors"
    exit 1
fi

# Check if required files exist
echo ""
echo "Checking required files..."
if [ -f "docker-compose.yml" ]; then
    echo "✓ docker-compose.yml exists"
else
    echo "❌ docker-compose.yml not found"
    exit 1
fi

if [ -f "prometheus.yml" ]; then
    echo "✓ prometheus.yml exists"
else
    echo "❌ prometheus.yml not found"
    exit 1
fi

if [ -f "Dockerfile.server" ]; then
    echo "✓ Dockerfile.server exists"
else
    echo "⚠ Dockerfile.server not found (will be needed for build)"
fi

# Check if .env file exists
echo ""
if [ -f ".env" ]; then
    echo "✓ .env file exists"
    echo ""
    echo "Environment variables configured:"
    grep -E "^MORGAN_" .env | sed 's/=.*/=***/' || true
else
    echo "⚠ .env file not found"
    echo "  Copy .env.example to .env and configure your settings"
fi

echo ""
echo "=== Validation Complete ==="
echo ""
echo "To start services:"
echo "  docker-compose up -d"
echo ""
echo "To start with monitoring:"
echo "  docker-compose --profile monitoring up -d"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f"
