#!/bin/bash
#
# Morgan AI Assistant - Setup and Deployment Script
#
# Usage:
#   ./setup.sh [OPTIONS] [COMPONENTS...]
#
# Examples:
#   ./setup.sh --core --db                    # Start core services with databases
#   ./setup.sh --all                          # Start all services
#   ./setup.sh --check-models                 # Check and download required Ollama models
#   ./setup.sh --core --env=./custom.env      # Use custom env file
#   ./setup.sh --stop                         # Stop all services
#   ./setup.sh --status                       # Show service status
#   ./setup.sh --logs core                    # View logs for core service
#
# Components:
#   --core        Morgan core server (API, orchestration)
#   --db          Databases (Qdrant + Redis)
#   --reranking   Reranking service (Docker-based)
#   --monitoring  Prometheus + Grafana
#   --background  Background job processor
#   --all         All components
#
# Ollama (baremetal - not Docker):
#   --check-models    Check if required models are installed
#   --pull-models     Download missing Ollama models
#   --ollama-status   Show Ollama service status
#
# Options:
#   --env=FILE    Specify environment file (default: .env)
#   --build       Force rebuild containers
#   --pull        Pull latest Docker images
#   --stop        Stop services
#   --restart     Restart services
#   --status      Show service status
#   --logs        View service logs
#   --distributed Use distributed compose file
#   -d, --detach  Run in background (default)
#   -f, --foreground  Run in foreground
#   -h, --help    Show this help message
#
# Copyright 2025 Morgan AI Assistant Contributors
# SPDX-License-Identifier: Apache-2.0

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="${SCRIPT_DIR}/docker"
CONFIG_DIR="${SCRIPT_DIR}/config"
DEFAULT_ENV_FILE="${SCRIPT_DIR}/.env"
COMPOSE_FILE="${DOCKER_DIR}/docker-compose.yml"
DISTRIBUTED_COMPOSE_FILE="${DOCKER_DIR}/docker-compose.distributed.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# Required Ollama Models (baremetal)
# =============================================================================

# Models that should be available on the Ollama server
REQUIRED_LLM_MODELS=(
    "qwen2.5:32b-instruct-q4_K_M"
    "qwen2.5:7b-instruct-q5_K_M"
)

REQUIRED_EMBEDDING_MODELS=(
    "qwen3-embedding:4b"
)

# Optional models
OPTIONAL_MODELS=(
    "qwen2.5:14b-instruct-q4_K_M"
)

# =============================================================================
# Default values
# =============================================================================

ENV_FILE="${DEFAULT_ENV_FILE}"
DETACH=true
BUILD=false
PULL=false
USE_DISTRIBUTED=false

# Ollama endpoint (baremetal)
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

# Component flags
COMP_CORE=false
COMP_DB=false
COMP_RERANKING=false
COMP_MONITORING=false
COMP_BACKGROUND=false
COMP_ALL=false

# Action flags
ACTION_START=true
ACTION_STOP=false
ACTION_RESTART=false
ACTION_STATUS=false
ACTION_LOGS=false
ACTION_CHECK_MODELS=false
ACTION_PULL_MODELS=false
ACTION_OLLAMA_STATUS=false
LOGS_SERVICE=""

# =============================================================================
# Functions
# =============================================================================

print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                                                              ║"
    echo "║   ███╗   ███╗ ██████╗ ██████╗  ██████╗  █████╗ ███╗   ██╗   ║"
    echo "║   ████╗ ████║██╔═══██╗██╔══██╗██╔════╝ ██╔══██╗████╗  ██║   ║"
    echo "║   ██╔████╔██║██║   ██║██████╔╝██║  ███╗███████║██╔██╗ ██║   ║"
    echo "║   ██║╚██╔╝██║██║   ██║██╔══██╗██║   ██║██╔══██║██║╚██╗██║   ║"
    echo "║   ██║ ╚═╝ ██║╚██████╔╝██║  ██║╚██████╔╝██║  ██║██║ ╚████║   ║"
    echo "║   ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ║"
    echo "║                                                              ║"
    echo "║              Personal AI Assistant - Self-Hosted             ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_help() {
    print_banner
    echo "Usage: $0 [OPTIONS] [COMPONENTS...]"
    echo ""
    echo -e "${GREEN}Docker Components:${NC}"
    echo "  --core        Morgan core server (API, orchestration)"
    echo "  --db          Databases (Qdrant vector DB + Redis cache)"
    echo "  --reranking   Reranking service (CrossEncoder)"
    echo "  --monitoring  Monitoring stack (Prometheus + Grafana)"
    echo "  --background  Background job processor"
    echo "  --all         All Docker components"
    echo ""
    echo -e "${GREEN}Ollama (Baremetal):${NC}"
    echo "  --check-models    Check if required models are installed"
    echo "  --pull-models     Download missing Ollama models"
    echo "  --ollama-status   Show Ollama service status"
    echo ""
    echo -e "${GREEN}Options:${NC}"
    echo "  --env=FILE       Specify environment file (default: .env)"
    echo "  --build          Force rebuild containers"
    echo "  --pull           Pull latest Docker images before starting"
    echo "  --distributed    Use distributed multi-host compose file"
    echo "  -d, --detach     Run in background (default)"
    echo "  -f, --foreground Run in foreground"
    echo ""
    echo -e "${GREEN}Actions:${NC}"
    echo "  --stop           Stop Docker services"
    echo "  --restart        Restart Docker services"
    echo "  --status         Show all service status"
    echo "  --logs [SERVICE] View service logs (optional service name)"
    echo ""
    echo -e "${GREEN}Examples:${NC}"
    echo "  $0 --check-models                 # Verify Ollama models"
    echo "  $0 --pull-models                  # Download missing models"
    echo "  $0 --core --db                    # Start core + databases"
    echo "  $0 --all                          # Start all Docker services"
    echo "  $0 --core --env=./prod.env        # Use production env file"
    echo "  $0 --stop                         # Stop all services"
    echo "  $0 --status                       # Show complete status"
    echo ""
    echo -e "${GREEN}Environment Variables:${NC}"
    echo "  OLLAMA_HOST      Ollama API endpoint (default: http://localhost:11434)"
    echo ""
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        log_info "Visit: https://docs.docker.com/engine/install/"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running or you don't have permission."
        log_info "Try: sudo systemctl start docker"
        log_info "Or: Add your user to the docker group"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed."
        log_info "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi
}

check_ollama() {
    log_info "Checking Ollama at: ${OLLAMA_HOST}"

    if ! curl -sf "${OLLAMA_HOST}/api/version" > /dev/null 2>&1; then
        log_error "Ollama is not running or not accessible at ${OLLAMA_HOST}"
        log_info "Start Ollama with: ollama serve"
        log_info "Or check OLLAMA_HOST environment variable"
        return 1
    fi

    local version=$(curl -sf "${OLLAMA_HOST}/api/version" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
    log_success "Ollama is running (version: ${version:-unknown})"
    return 0
}

list_ollama_models() {
    curl -sf "${OLLAMA_HOST}/api/tags" 2>/dev/null | grep -o '"name":"[^"]*"' | cut -d'"' -f4
}

check_model_installed() {
    local model="$1"
    local installed_models=$(list_ollama_models)

    # Check exact match or base name match
    if echo "$installed_models" | grep -qE "^${model}$|^${model}:"; then
        return 0
    fi

    # Check if model name without tag is installed with any tag
    local base_name=$(echo "$model" | cut -d':' -f1)
    if echo "$installed_models" | grep -q "^${base_name}:"; then
        return 0
    fi

    return 1
}

do_check_models() {
    log_info "Checking required Ollama models..."
    echo ""

    if ! check_ollama; then
        exit 1
    fi

    local missing_count=0
    local installed_count=0

    echo -e "${CYAN}LLM Models:${NC}"
    for model in "${REQUIRED_LLM_MODELS[@]}"; do
        if check_model_installed "$model"; then
            echo -e "  ${GREEN}✓${NC} $model"
            ((installed_count++))
        else
            echo -e "  ${RED}✗${NC} $model ${YELLOW}(missing)${NC}"
            ((missing_count++))
        fi
    done

    echo ""
    echo -e "${CYAN}Embedding Models:${NC}"
    for model in "${REQUIRED_EMBEDDING_MODELS[@]}"; do
        if check_model_installed "$model"; then
            echo -e "  ${GREEN}✓${NC} $model"
            ((installed_count++))
        else
            echo -e "  ${RED}✗${NC} $model ${YELLOW}(missing)${NC}"
            ((missing_count++))
        fi
    done

    echo ""
    echo -e "${CYAN}Optional Models:${NC}"
    for model in "${OPTIONAL_MODELS[@]}"; do
        if check_model_installed "$model"; then
            echo -e "  ${GREEN}✓${NC} $model"
        else
            echo -e "  ${YELLOW}○${NC} $model (optional, not installed)"
        fi
    done

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "Installed: ${GREEN}${installed_count}${NC}  Missing: ${RED}${missing_count}${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ $missing_count -gt 0 ]; then
        echo ""
        log_warn "Some required models are missing."
        log_info "Run: $0 --pull-models  to download them"
        return 1
    else
        log_success "All required models are installed!"
        return 0
    fi
}

do_pull_models() {
    log_info "Downloading required Ollama models..."
    echo ""

    if ! check_ollama; then
        exit 1
    fi

    local total_models=0
    local downloaded=0
    local skipped=0
    local failed=0

    # Combine required models
    local all_models=("${REQUIRED_LLM_MODELS[@]}" "${REQUIRED_EMBEDDING_MODELS[@]}")
    total_models=${#all_models[@]}

    for model in "${all_models[@]}"; do
        echo ""
        if check_model_installed "$model"; then
            log_info "Model already installed: $model"
            ((skipped++))
        else
            log_info "Downloading: $model"
            echo "  This may take a while depending on model size..."

            if ollama pull "$model"; then
                log_success "Downloaded: $model"
                ((downloaded++))
            else
                log_error "Failed to download: $model"
                ((failed++))
            fi
        fi
    done

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "Total: ${total_models}  Downloaded: ${GREEN}${downloaded}${NC}  Skipped: ${YELLOW}${skipped}${NC}  Failed: ${RED}${failed}${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ $failed -gt 0 ]; then
        log_error "Some models failed to download"
        return 1
    fi

    log_success "Model download complete!"
}

do_ollama_status() {
    log_info "Ollama Service Status"
    echo ""

    if check_ollama; then
        echo ""
        echo -e "${CYAN}Installed Models:${NC}"
        local models=$(list_ollama_models)
        if [ -n "$models" ]; then
            echo "$models" | while read -r model; do
                echo "  • $model"
            done
        else
            echo "  (no models installed)"
        fi

        echo ""
        echo -e "${CYAN}Running Models:${NC}"
        local running=$(curl -sf "${OLLAMA_HOST}/api/ps" 2>/dev/null)
        if echo "$running" | grep -q '"models":\[\]'; then
            echo "  (no models currently loaded)"
        else
            echo "$running" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | while read -r model; do
                echo "  • $model (loaded)"
            done
        fi
    fi
}

load_env_file() {
    if [ -f "$ENV_FILE" ]; then
        log_info "Loading environment from: $ENV_FILE"
        set -a
        source "$ENV_FILE"
        set +a

        # Update OLLAMA_HOST if set in env file
        if [ -n "${MORGAN_LLM_ENDPOINT:-}" ]; then
            OLLAMA_HOST="${MORGAN_LLM_ENDPOINT}"
        fi
    else
        if [ "$ENV_FILE" != "$DEFAULT_ENV_FILE" ]; then
            log_error "Environment file not found: $ENV_FILE"
            exit 1
        else
            log_warn "No .env file found. Using default configuration."
            log_info "Copy .env.example to .env and configure as needed."
        fi
    fi
}

get_compose_cmd() {
    if docker compose version &> /dev/null 2>&1; then
        echo "docker compose"
    else
        echo "docker-compose"
    fi
}

build_compose_args() {
    local compose_cmd=$(get_compose_cmd)
    local args=""
    local services=""
    local profiles=""

    # Select compose file
    if [ "$USE_DISTRIBUTED" = true ]; then
        args="-f ${DISTRIBUTED_COMPOSE_FILE}"
    else
        args="-f ${COMPOSE_FILE}"
    fi

    # Add env file
    if [ -f "$ENV_FILE" ]; then
        args="${args} --env-file ${ENV_FILE}"
    fi

    # Build profiles and services list based on components
    if [ "$COMP_ALL" = true ]; then
        services=""  # Empty means all services
        profiles="--profile monitoring --profile background"
    else
        if [ "$COMP_CORE" = true ]; then
            services="${services} morgan-server"
            if [ "$USE_DISTRIBUTED" = true ]; then
                services="${services} morgan-core"
            fi
        fi

        if [ "$COMP_DB" = true ]; then
            services="${services} qdrant redis"
        fi

        if [ "$COMP_RERANKING" = true ]; then
            if [ "$USE_DISTRIBUTED" = true ]; then
                profiles="${profiles} --profile gpu-reranking"
            fi
        fi

        if [ "$COMP_MONITORING" = true ]; then
            profiles="${profiles} --profile monitoring"
            services="${services} prometheus grafana"
        fi

        if [ "$COMP_BACKGROUND" = true ]; then
            profiles="${profiles} --profile background"
            if [ "$USE_DISTRIBUTED" = true ]; then
                services="${services} morgan-background"
            fi
        fi
    fi

    echo "${compose_cmd} ${args} ${profiles} ${services}"
}

do_start() {
    log_info "Starting Morgan Docker services..."

    # First check if Ollama is available
    echo ""
    if check_ollama; then
        echo ""
        # Quick model check
        local missing=false
        for model in "${REQUIRED_LLM_MODELS[@]}" "${REQUIRED_EMBEDDING_MODELS[@]}"; do
            if ! check_model_installed "$model"; then
                missing=true
                break
            fi
        done

        if [ "$missing" = true ]; then
            log_warn "Some required Ollama models are missing."
            log_info "Run: $0 --check-models  to see details"
            log_info "Run: $0 --pull-models   to download them"
            echo ""
        fi
    else
        log_warn "Ollama is not running. LLM features will not work."
        echo ""
    fi

    local compose_args=$(build_compose_args)
    local up_args="up"

    if [ "$DETACH" = true ]; then
        up_args="${up_args} -d"
    fi

    if [ "$BUILD" = true ]; then
        up_args="${up_args} --build"
    fi

    if [ "$PULL" = true ]; then
        log_info "Pulling latest Docker images..."
        eval "${compose_args} pull"
    fi

    log_info "Running: ${compose_args} ${up_args}"
    eval "${compose_args} ${up_args}"

    if [ "$DETACH" = true ]; then
        log_success "Docker services started in background."
        echo ""
        do_status
    fi
}

do_stop() {
    log_info "Stopping Morgan Docker services..."

    local compose_cmd=$(get_compose_cmd)
    local compose_file="${COMPOSE_FILE}"

    if [ "$USE_DISTRIBUTED" = true ]; then
        compose_file="${DISTRIBUTED_COMPOSE_FILE}"
    fi

    ${compose_cmd} -f "${compose_file}" down

    log_success "Docker services stopped."
}

do_restart() {
    log_info "Restarting Morgan Docker services..."
    do_stop
    sleep 2
    do_start
}

do_status() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}                    Morgan Service Status                       ${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Ollama Status
    echo -e "${YELLOW}Ollama (Baremetal):${NC}"
    if curl -sf "${OLLAMA_HOST}/api/version" > /dev/null 2>&1; then
        echo -e "  Status:   ${GREEN}RUNNING${NC}"
        echo -e "  Endpoint: ${OLLAMA_HOST}"

        # Count models
        local model_count=$(list_ollama_models | wc -l)
        echo -e "  Models:   ${model_count} installed"
    else
        echo -e "  Status:   ${RED}NOT RUNNING${NC}"
        echo -e "  Endpoint: ${OLLAMA_HOST}"
    fi

    echo ""
    echo -e "${YELLOW}Docker Services:${NC}"

    local compose_cmd=$(get_compose_cmd)
    local compose_file="${COMPOSE_FILE}"

    if [ "$USE_DISTRIBUTED" = true ]; then
        compose_file="${DISTRIBUTED_COMPOSE_FILE}"
    fi

    ${compose_cmd} -f "${compose_file}" ps 2>/dev/null || echo "  (no services running)"

    echo ""
    echo -e "${YELLOW}Health Checks:${NC}"

    # Check core services
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "morgan-server\|morgan-core"; then
        if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
            echo -e "  Morgan Server: ${GREEN}HEALTHY${NC}"
        else
            echo -e "  Morgan Server: ${RED}UNHEALTHY${NC}"
        fi
    fi

    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "qdrant"; then
        if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
            echo -e "  Qdrant:        ${GREEN}HEALTHY${NC}"
        else
            echo -e "  Qdrant:        ${RED}UNHEALTHY${NC}"
        fi
    fi

    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "redis"; then
        if docker exec morgan-redis redis-cli ping > /dev/null 2>&1; then
            echo -e "  Redis:         ${GREEN}HEALTHY${NC}"
        else
            echo -e "  Redis:         ${RED}UNHEALTHY${NC}"
        fi
    fi

    echo ""
}

do_logs() {
    local compose_cmd=$(get_compose_cmd)
    local compose_file="${COMPOSE_FILE}"

    if [ "$USE_DISTRIBUTED" = true ]; then
        compose_file="${DISTRIBUTED_COMPOSE_FILE}"
    fi

    if [ -n "$LOGS_SERVICE" ]; then
        log_info "Showing logs for: ${LOGS_SERVICE}"
        ${compose_cmd} -f "${compose_file}" logs -f "${LOGS_SERVICE}"
    else
        log_info "Showing logs for all services (Ctrl+C to exit)"
        ${compose_cmd} -f "${compose_file}" logs -f
    fi
}

create_env_example() {
    if [ ! -f "${SCRIPT_DIR}/.env.example" ]; then
        cat > "${SCRIPT_DIR}/.env.example" << 'EOF'
# Morgan AI Assistant - Environment Configuration
# Copy this file to .env and customize as needed

# =============================================================================
# Ollama Configuration (Baremetal - not Docker)
# =============================================================================
# Ollama endpoint - adjust if running on different host
MORGAN_LLM_ENDPOINT=http://localhost:11434
MORGAN_LLM_MODEL=qwen2.5:32b-instruct-q4_K_M
MORGAN_LLM_FAST_MODEL=qwen2.5:7b-instruct-q5_K_M

# For distributed setup (comma-separated endpoints)
# MORGAN_LLM_ENDPOINTS=http://192.168.1.20:11434/v1,http://192.168.1.21:11434/v1
MORGAN_LLM_STRATEGY=round_robin

# =============================================================================
# Embedding Configuration (via Ollama)
# =============================================================================
MORGAN_EMBEDDING_PROVIDER=ollama
MORGAN_EMBEDDING_MODEL=qwen3-embedding:4b
MORGAN_EMBEDDING_DIMENSIONS=2048
# MORGAN_EMBEDDING_ENDPOINT=http://192.168.1.22:11434/v1

# =============================================================================
# Reranking Configuration
# =============================================================================
MORGAN_RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
# MORGAN_RERANKING_ENDPOINT=http://192.168.1.23:8080/rerank

# =============================================================================
# Database Configuration (Docker)
# =============================================================================
# Redis
MORGAN_REDIS_PASSWORD=
MORGAN_REDIS_PREFIX=morgan:

# Qdrant
MORGAN_VECTOR_DB_API_KEY=

# =============================================================================
# Model Downloads (Optional)
# =============================================================================
# Hugging Face token for gated models
HF_TOKEN=

# =============================================================================
# Monitoring (Optional)
# =============================================================================
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=CHANGE_ME_IN_PRODUCTION

# =============================================================================
# Performance
# =============================================================================
MORGAN_LOG_LEVEL=INFO
MORGAN_LOG_FORMAT=json
MORGAN_CACHE_SIZE_MB=1000
MORGAN_MAX_CONCURRENT=100
MORGAN_REQUEST_TIMEOUT=60
EOF
        log_info "Created .env.example file"
    fi
}

# =============================================================================
# Main
# =============================================================================

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --core)
            COMP_CORE=true
            shift
            ;;
        --db)
            COMP_DB=true
            shift
            ;;
        --reranking)
            COMP_RERANKING=true
            shift
            ;;
        --monitoring)
            COMP_MONITORING=true
            shift
            ;;
        --background)
            COMP_BACKGROUND=true
            shift
            ;;
        --all)
            COMP_ALL=true
            shift
            ;;
        --env=*)
            ENV_FILE="${1#*=}"
            shift
            ;;
        --build)
            BUILD=true
            shift
            ;;
        --pull)
            PULL=true
            shift
            ;;
        --distributed)
            USE_DISTRIBUTED=true
            shift
            ;;
        -d|--detach)
            DETACH=true
            shift
            ;;
        -f|--foreground)
            DETACH=false
            shift
            ;;
        --stop)
            ACTION_START=false
            ACTION_STOP=true
            shift
            ;;
        --restart)
            ACTION_START=false
            ACTION_RESTART=true
            shift
            ;;
        --status)
            ACTION_START=false
            ACTION_STATUS=true
            shift
            ;;
        --logs)
            ACTION_START=false
            ACTION_LOGS=true
            shift
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                LOGS_SERVICE="$1"
                shift
            fi
            ;;
        --check-models)
            ACTION_START=false
            ACTION_CHECK_MODELS=true
            shift
            ;;
        --pull-models)
            ACTION_START=false
            ACTION_PULL_MODELS=true
            shift
            ;;
        --ollama-status)
            ACTION_START=false
            ACTION_OLLAMA_STATUS=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Show banner
print_banner

# Create .env.example if needed
create_env_example

# Load environment
load_env_file

# Execute action
if [ "$ACTION_CHECK_MODELS" = true ]; then
    do_check_models
elif [ "$ACTION_PULL_MODELS" = true ]; then
    do_pull_models
elif [ "$ACTION_OLLAMA_STATUS" = true ]; then
    do_ollama_status
elif [ "$ACTION_STOP" = true ]; then
    check_docker
    do_stop
elif [ "$ACTION_RESTART" = true ]; then
    check_docker
    do_restart
elif [ "$ACTION_STATUS" = true ]; then
    do_status
elif [ "$ACTION_LOGS" = true ]; then
    check_docker
    do_logs
elif [ "$ACTION_START" = true ]; then
    check_docker

    # Check if at least one component selected
    if [ "$COMP_CORE" = false ] && [ "$COMP_DB" = false ] && \
       [ "$COMP_RERANKING" = false ] && \
       [ "$COMP_MONITORING" = false ] && [ "$COMP_BACKGROUND" = false ] && \
       [ "$COMP_ALL" = false ]; then
        log_warn "No components specified. Use --help for options."
        log_info "Starting default components: --core --db"
        COMP_CORE=true
        COMP_DB=true
    fi

    do_start
fi
