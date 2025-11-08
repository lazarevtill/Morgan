#!/usr/bin/env bash
#
# Morgan v2-0.0.1 - Complete Development Setup Script
#
# This script performs a complete one-command setup of the Morgan development environment.
# It is idempotent - can be run multiple times safely.
#
# Usage:
#   ./scripts/setup.sh [OPTIONS]
#
# Options:
#   --skip-uv          Skip uv installation
#   --skip-docker      Skip Docker setup
#   --skip-hooks       Skip pre-commit hooks installation
#   --production       Setup for production (minimal dev dependencies)
#   --help             Show this help message
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
SKIP_UV=false
SKIP_DOCKER=false
SKIP_HOOKS=false
PRODUCTION=false

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show help
show_help() {
    grep '^#' "$0" | grep -v '#!/usr/bin/env' | sed 's/^# //g' | sed 's/^#//g'
    exit 0
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-uv)
                SKIP_UV=true
                shift
                ;;
            --skip-docker)
                SKIP_DOCKER=true
                shift
                ;;
            --skip-hooks)
                SKIP_HOOKS=true
                shift
                ;;
            --production)
                PRODUCTION=true
                shift
                ;;
            --help)
                show_help
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                ;;
        esac
    done
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."

    # Check Python
    if ! command_exists python3; then
        log_error "Python 3 is not installed. Please install Python 3.11 or later."
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    log_info "Python version: $PYTHON_VERSION"

    # Check Docker (if not skipped)
    if [[ "$SKIP_DOCKER" == false ]]; then
        if ! command_exists docker; then
            log_warning "Docker is not installed. Docker features will be skipped."
            SKIP_DOCKER=true
        else
            DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
            log_info "Docker version: $DOCKER_VERSION"
        fi
    fi

    # Check for nvidia-smi (GPU support)
    if command_exists nvidia-smi; then
        log_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=gpu_name,driver_version --format=csv,noheader | head -1
    else
        log_warning "No NVIDIA GPU detected. GPU-accelerated services will not be available."
    fi

    log_success "System requirements check completed"
}

# Install and configure NetBird VPN
setup_netbird() {
    log_info "Setting up NetBird VPN (required for private Nexus access)..."

    # Check if NetBird is already installed
    if command_exists netbird; then
        log_success "NetBird is already installed"
    else
        log_info "Installing NetBird..."
        curl -fsSL https://pkgs.netbird.io/install.sh | sh

        if command_exists netbird; then
            log_success "NetBird installed successfully"
        else
            log_error "Failed to install NetBird"
            log_error "Please install NetBird manually: https://netbird.io/docs/getting-started/installation"
            exit 1
        fi
    fi

    # Check if already connected
    if sudo netbird status 2>/dev/null | grep -q "Connected"; then
        log_success "NetBird is already connected"
        return
    fi

    # Prompt for setup key if not in environment
    if [[ -z "${NETBIRD_SETUP_KEY:-}" ]]; then
        log_warning "NetBird setup key not found in environment"
        echo ""
        echo "Please enter your NetBird setup key (or press Enter to skip):"
        echo "You can get the setup key from your team administrator"
        read -r NETBIRD_SETUP_KEY

        if [[ -z "$NETBIRD_SETUP_KEY" ]]; then
            log_warning "Skipping NetBird connection. You will need to connect manually later."
            log_info "To connect later, run: sudo netbird up --management-url https://vpn.lazarev.cloud --setup-key <your-key>"
            return
        fi
    fi

    # Ensure NetBird service is running
    log_info "Starting NetBird service..."
    sudo netbird service install 2>/dev/null || true
    sudo netbird service start 2>/dev/null || true

    # Wait for daemon to be ready
    sleep 2

    # Connect to NetBird
    log_info "Connecting to NetBird VPN..."
    if sudo netbird up \
        --management-url https://vpn.lazarev.cloud \
        --setup-key "$NETBIRD_SETUP_KEY" \
        --log-level info; then
        log_success "Successfully connected to NetBird VPN"

        # Wait for connection to stabilize
        sleep 3

        # Verify connection
        if sudo netbird status 2>/dev/null | grep -q "Connected"; then
            log_success "NetBird connection verified"
        else
            log_warning "NetBird connection may not be stable yet"
        fi
    else
        log_error "Failed to connect to NetBird VPN"
        log_error "Please connect manually: sudo netbird up --management-url https://vpn.lazarev.cloud --setup-key <your-key>"
        exit 1
    fi
}

# Install uv package manager
install_uv() {
    if [[ "$SKIP_UV" == true ]]; then
        log_info "Skipping uv installation"
        return
    fi

    log_info "Setting up uv package manager..."

    if command_exists uv; then
        log_success "uv is already installed: $(uv --version)"
    else
        log_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Add to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"

        if command_exists uv; then
            log_success "uv installed successfully: $(uv --version)"
        else
            log_error "Failed to install uv. Please install manually."
            exit 1
        fi
    fi

    # Configure uv for Nexus proxy
    log_info "Configuring uv with Nexus proxy..."
    export UV_INDEX_URL="https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple"
    echo 'export UV_INDEX_URL="https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple"' >> ~/.bashrc || true
}

# Create virtual environment
setup_venv() {
    log_info "Setting up Python virtual environment..."

    cd "$PROJECT_ROOT"

    if [[ -d ".venv" ]]; then
        log_info "Virtual environment already exists"
    else
        log_info "Creating virtual environment..."
        python3 -m venv .venv
        log_success "Virtual environment created"
    fi

    # Activate virtual environment
    source .venv/bin/activate

    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip wheel setuptools
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."

    cd "$PROJECT_ROOT"
    source .venv/bin/activate

    if command_exists uv; then
        log_info "Installing dependencies with uv..."
        uv pip install -r requirements-base.txt
        uv pip install -r requirements-core.txt

        if [[ "$PRODUCTION" == false ]]; then
            log_info "Installing development dependencies..."
            uv pip install pytest pytest-cov pytest-asyncio pytest-mock httpx \
                black isort ruff mypy types-PyYAML types-redis \
                pre-commit bandit safety pip-licenses
        fi
    else
        log_info "Installing dependencies with pip..."
        pip install -r requirements-base.txt
        pip install -r requirements-core.txt

        if [[ "$PRODUCTION" == false ]]; then
            pip install pytest pytest-cov pytest-asyncio pytest-mock httpx \
                black isort ruff mypy types-PyYAML types-redis \
                pre-commit bandit safety pip-licenses
        fi
    fi

    log_success "Dependencies installed successfully"
}

# Setup pre-commit hooks
setup_precommit() {
    if [[ "$SKIP_HOOKS" == true ]] || [[ "$PRODUCTION" == true ]]; then
        log_info "Skipping pre-commit hooks setup"
        return
    fi

    log_info "Setting up pre-commit hooks..."

    cd "$PROJECT_ROOT"
    source .venv/bin/activate

    if command_exists pre-commit; then
        pre-commit install
        pre-commit install --hook-type commit-msg
        log_success "Pre-commit hooks installed"
    else
        log_warning "pre-commit not found. Install it with: pip install pre-commit"
    fi
}

# Setup environment files
setup_env_files() {
    log_info "Setting up environment files..."

    cd "$PROJECT_ROOT"

    if [[ ! -f ".env" ]]; then
        if [[ -f ".env.example" ]]; then
            cp .env.example .env
            log_success "Created .env file from .env.example"
            log_warning "Please edit .env file with your configuration"
        else
            log_warning ".env.example not found, skipping .env creation"
        fi
    else
        log_info ".env file already exists"
    fi
}

# Setup Docker environment
setup_docker() {
    if [[ "$SKIP_DOCKER" == true ]]; then
        log_info "Skipping Docker setup"
        return
    fi

    log_info "Setting up Docker environment..."

    cd "$PROJECT_ROOT"

    # Create necessary directories
    log_info "Creating data directories..."
    mkdir -p data/models/{torch_hub,transformers,huggingface,tts,stt,faster_whisper,sentence_transformers}
    mkdir -p data/voices
    mkdir -p logs/{core,llm,tts,stt}

    log_success "Data directories created"

    # Build Docker images
    log_info "Building Docker images (this may take a while)..."
    docker compose build --pull || {
        log_error "Docker build failed"
        return 1
    }

    log_success "Docker images built successfully"
}

# Initialize databases
init_databases() {
    if [[ "$SKIP_DOCKER" == true ]]; then
        log_info "Skipping database initialization"
        return
    fi

    log_info "Initializing databases..."

    cd "$PROJECT_ROOT"

    # Start database services
    log_info "Starting database services..."
    docker compose up -d postgres redis qdrant

    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 10

    # Check if databases are ready
    if docker compose exec -T postgres pg_isready -U morgan >/dev/null 2>&1; then
        log_success "PostgreSQL is ready"
    else
        log_warning "PostgreSQL may not be ready yet"
    fi

    if docker compose exec -T redis redis-cli ping >/dev/null 2>&1; then
        log_success "Redis is ready"
    else
        log_warning "Redis may not be ready yet"
    fi

    log_success "Database initialization completed"
}

# Run tests
run_tests() {
    if [[ "$PRODUCTION" == true ]]; then
        log_info "Skipping tests in production mode"
        return
    fi

    log_info "Running basic tests..."

    cd "$PROJECT_ROOT"
    source .venv/bin/activate

    # Run a quick test to verify setup
    if pytest tests/ -v --tb=short -m "not integration and not slow" --maxfail=1 >/dev/null 2>&1; then
        log_success "Basic tests passed"
    else
        log_warning "Some tests failed. This is okay for initial setup."
    fi
}

# Print summary
print_summary() {
    echo ""
    log_success "================================"
    log_success "Setup completed successfully!"
    log_success "================================"
    echo ""
    log_info "Next steps:"
    echo ""
    echo "1. Activate virtual environment:"
    echo "   source .venv/bin/activate"
    echo ""
    echo "2. Configure your environment:"
    echo "   Edit .env file with your settings"
    echo ""
    echo "3. Start services:"
    echo "   docker compose up -d"
    echo ""
    echo "4. Run tests:"
    echo "   make test"
    echo ""
    echo "5. Start development:"
    echo "   make run"
    echo ""
    log_info "For more information, see docs/getting-started/DEVELOPMENT.md"
    echo ""
}

# Main function
main() {
    echo ""
    log_info "================================"
    log_info "Morgan v2-0.0.1 Setup"
    log_info "================================"
    echo ""

    parse_args "$@"

    check_requirements
    setup_netbird
    install_uv
    setup_venv
    install_dependencies
    setup_precommit
    setup_env_files
    setup_docker
    init_databases
    run_tests

    print_summary
}

# Run main function
main "$@"
