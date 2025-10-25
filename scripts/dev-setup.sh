#!/bin/bash

# Morgan AI Assistant - Development Setup Script
# This script sets up the development environment using uv for local development

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Setting up Morgan AI Assistant development environment...${NC}"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}📦 Installing uv (fast Python package manager)...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Configure uv to use Nexus PyPI proxy
echo -e "${BLUE}🔧 Configuring uv to use Nexus PyPI proxy...${NC}"
export UV_INDEX_URL="https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple"

# Create necessary directories
echo -e "${BLUE}📁 Creating necessary directories...${NC}"
mkdir -p logs/{llm,tts,stt,vad,core}
mkdir -p data/{models/{llm,tts,stt},voices,conversations}

# Create virtual environment for local development (system Python is protected)
echo -e "${BLUE}📦 Creating virtual environment for local development...${NC}"
uv venv

# Activate virtual environment and install dependencies
echo -e "${BLUE}📦 Activating virtual environment and installing dependencies...${NC}"
source .venv/bin/activate
uv pip install fastapi uvicorn[standard] pydantic aiohttp pyyaml python-dotenv structlog psutil redis

# Install development dependencies
echo -e "${BLUE}📦 Installing development dependencies...${NC}"
uv pip install pytest pytest-asyncio httpx pytest-cov black isort flake8

# Validate repository configurations
echo -e "${BLUE}🔍 Validating repository configurations...${NC}"
if [ -f "./scripts/fix-repositories.sh" ]; then
    if ./scripts/fix-repositories.sh -v &>/dev/null; then
        echo -e "${GREEN}✓ Repository configurations are valid${NC}"
    else
        echo -e "${YELLOW}⚠️  Repository configurations need attention${NC}"
        echo -e "${BLUE}Run: ./scripts/fix-repositories.sh -f${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Repository validation script not found${NC}"
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${GREEN}📝 Creating .env template file...${NC}"
    cat > .env << EOF
# Morgan AI Assistant Environment Configuration
# Copy this file to .env and modify as needed

# Ollama Configuration (external service)
OLLAMA_HOST=192.168.101.3:11434

# Service Configuration
MORGAN_CONFIG_DIR=./config

# CUDA Configuration
CUDA_VISIBLE_DEVICES=0

# Logging
LOG_LEVEL=INFO

# External Services (uncomment if needed)
# REDIS_URL=redis://localhost:6379
# POSTGRES_URL=postgresql://morgan:morgan_password@localhost:5432/morgan
EOF
    echo -e "${YELLOW}⚠️  Please edit .env file with your specific configuration${NC}"
fi

echo -e "${GREEN}✅ Development environment setup complete!${NC}"
echo ""
echo -e "${BLUE}📚 Available commands:${NC}"
echo -e "  ${GREEN}source .venv/bin/activate && python core/app.py${NC}              # Run core service locally"
echo -e "  ${GREEN}source .venv/bin/activate && python services/llm/main.py${NC}     # Run LLM service locally"
echo -e "  ${GREEN}source .venv/bin/activate && python services/tts/main.py${NC}     # Run TTS service locally"
echo -e "  ${GREEN}source .venv/bin/activate && python services/stt/main.py${NC}     # Run STT service locally"
echo -e "  ${GREEN}source .venv/bin/activate && python services/vad/main.py${NC}     # Run VAD service locally"
echo -e "  ${GREEN}source .venv/bin/activate && pytest${NC}                         # Run tests"
echo -e "  ${GREEN}docker-compose up -d${NC}                                        # Start all services in Docker"
echo -e "  ${GREEN}docker-compose build${NC}                                        # Build all Docker images"
echo -e "  ${GREEN}./scripts/dev-setup.ps1 -Build -Up${NC}                          # Windows PowerShell setup"
echo ""
echo -e "${BLUE}💡 Notes:${NC}"
echo -e "  • Virtual environment created for local development (.venv)"
echo -e "  • Docker containers use system Python (no venv in containers)"
echo -e "  • Docker containers use Nexus proxy repositories for faster builds"
echo -e "  • External Ollama service should be running at 192.168.101.3:11434"
echo -e "  • CUDA services require NVIDIA Container Toolkit"
echo -e "  • Use 'docker-compose logs -f <service>' to view service logs"
echo ""
echo -e "${YELLOW}🔗 Nexus Repositories Configured:${NC}"
echo -e "  • Ubuntu/Debian: https://nexus.in.lazarev.cloud/repository/ubuntu-group/"
echo -e "  • Security: https://nexus.in.lazarev.cloud/repository/debian-security/"
echo -e "  • PyPI: https://nexus.in.lazarev.cloud/repository/pypi-proxy/"