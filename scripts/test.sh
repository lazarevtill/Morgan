#!/usr/bin/env bash
#
# Morgan v2-0.0.1 - Comprehensive Test Runner Script
#
# This script runs all tests with various options and configurations.
# It is idempotent and can be run multiple times safely.
#
# Usage:
#   ./scripts/test.sh [OPTIONS]
#
# Options:
#   --unit             Run only unit tests (default)
#   --integration      Run only integration tests
#   --all              Run all tests
#   --coverage         Generate coverage report
#   --watch            Watch mode (re-run tests on file changes)
#   --verbose          Verbose output
#   --fast             Skip slow tests
#   --parallel         Run tests in parallel
#   --failed           Re-run only failed tests
#   --docker           Run tests in Docker containers
#   --services         Test all microservices
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
TEST_TYPE="unit"
COVERAGE=false
WATCH=false
VERBOSE=false
FAST=false
PARALLEL=false
FAILED=false
DOCKER=false
SERVICES=false

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
            --unit)
                TEST_TYPE="unit"
                shift
                ;;
            --integration)
                TEST_TYPE="integration"
                shift
                ;;
            --all)
                TEST_TYPE="all"
                shift
                ;;
            --coverage)
                COVERAGE=true
                shift
                ;;
            --watch)
                WATCH=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --fast)
                FAST=true
                shift
                ;;
            --parallel)
                PARALLEL=true
                shift
                ;;
            --failed)
                FAILED=true
                shift
                ;;
            --docker)
                DOCKER=true
                shift
                ;;
            --services)
                SERVICES=true
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

# Check if virtual environment is activated
check_venv() {
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        log_warning "Virtual environment not activated"
        if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
            log_info "Activating virtual environment..."
            source "$PROJECT_ROOT/.venv/bin/activate"
        else
            log_error "Virtual environment not found. Run ./scripts/setup.sh first."
            exit 1
        fi
    fi
}

# Setup test environment
setup_test_env() {
    log_info "Setting up test environment..."

    cd "$PROJECT_ROOT"

    # Export test environment variables
    export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
    export POSTGRES_PORT="${POSTGRES_PORT:-5432}"
    export POSTGRES_DB="${POSTGRES_DB:-morgan_test}"
    export POSTGRES_USER="${POSTGRES_USER:-morgan}"
    export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-test_password}"
    export REDIS_HOST="${REDIS_HOST:-localhost}"
    export REDIS_PORT="${REDIS_PORT:-6379}"
    export QDRANT_HOST="${QDRANT_HOST:-localhost}"
    export QDRANT_PORT="${QDRANT_PORT:-6333}"
    export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

    log_success "Test environment configured"
}

# Start required services for integration tests
start_services() {
    if [[ "$TEST_TYPE" == "integration" ]] || [[ "$TEST_TYPE" == "all" ]]; then
        log_info "Starting required services for integration tests..."

        if command -v docker >/dev/null 2>&1; then
            docker compose up -d postgres redis qdrant

            log_info "Waiting for services to be ready..."
            sleep 5

            # Wait for PostgreSQL
            for i in {1..30}; do
                if docker compose exec -T postgres pg_isready -U morgan >/dev/null 2>&1; then
                    log_success "PostgreSQL is ready"
                    break
                fi
                sleep 1
            done

            # Wait for Redis
            if docker compose exec -T redis redis-cli ping >/dev/null 2>&1; then
                log_success "Redis is ready"
            fi
        else
            log_warning "Docker not available. Integration tests may fail."
        fi
    fi
}

# Build pytest command
build_pytest_command() {
    local cmd="pytest"

    # Add verbosity
    if [[ "$VERBOSE" == true ]]; then
        cmd="$cmd -vv"
    else
        cmd="$cmd -v"
    fi

    # Add test markers
    if [[ "$TEST_TYPE" == "unit" ]]; then
        cmd="$cmd -m 'not integration and not slow'"
    elif [[ "$TEST_TYPE" == "integration" ]]; then
        cmd="$cmd -m 'integration'"
    fi

    # Skip slow tests if fast mode
    if [[ "$FAST" == true ]]; then
        cmd="$cmd -m 'not slow'"
    fi

    # Add coverage options
    if [[ "$COVERAGE" == true ]]; then
        cmd="$cmd --cov=core --cov=services --cov=shared"
        cmd="$cmd --cov-report=term-missing"
        cmd="$cmd --cov-report=html"
        cmd="$cmd --cov-report=xml"
    fi

    # Add parallel execution
    if [[ "$PARALLEL" == true ]]; then
        cmd="$cmd -n auto"
    fi

    # Re-run failed tests only
    if [[ "$FAILED" == true ]]; then
        cmd="$cmd --lf"
    fi

    # Add test directories
    cmd="$cmd tests/"

    # Ignore manual tests
    cmd="$cmd --ignore=tests/manual/"

    echo "$cmd"
}

# Run unit tests
run_unit_tests() {
    log_info "Running unit tests..."

    cd "$PROJECT_ROOT"

    local cmd=$(build_pytest_command)
    log_info "Command: $cmd"

    if eval "$cmd"; then
        log_success "Unit tests passed!"
        return 0
    else
        log_error "Unit tests failed!"
        return 1
    fi
}

# Run integration tests
run_integration_tests() {
    log_info "Running integration tests..."

    start_services

    cd "$PROJECT_ROOT"

    local cmd=$(build_pytest_command)
    log_info "Command: $cmd"

    if eval "$cmd"; then
        log_success "Integration tests passed!"
        return 0
    else
        log_error "Integration tests failed!"
        return 1
    fi
}

# Run all tests
run_all_tests() {
    log_info "Running all tests..."

    local unit_result=0
    local integration_result=0

    TEST_TYPE="unit"
    run_unit_tests || unit_result=$?

    TEST_TYPE="integration"
    run_integration_tests || integration_result=$?

    if [[ $unit_result -eq 0 ]] && [[ $integration_result -eq 0 ]]; then
        log_success "All tests passed!"
        return 0
    else
        log_error "Some tests failed!"
        return 1
    fi
}

# Run tests in Docker
run_docker_tests() {
    log_info "Running tests in Docker..."

    cd "$PROJECT_ROOT"

    docker compose run --rm -e PYTHONPATH=/app core pytest tests/ -v --tb=short

    log_success "Docker tests completed"
}

# Test individual services
test_services() {
    log_info "Testing microservices..."

    cd "$PROJECT_ROOT"

    # Start all services
    docker compose up -d

    log_info "Waiting for services to be ready..."
    sleep 30

    # Test Core service
    log_info "Testing Core service..."
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        log_success "Core service is healthy"
    else
        log_error "Core service health check failed"
    fi

    # Test LLM service
    log_info "Testing LLM service..."
    if curl -f http://localhost:8001/health >/dev/null 2>&1; then
        log_success "LLM service is healthy"
    else
        log_error "LLM service health check failed"
    fi

    # Test TTS service
    log_info "Testing TTS service..."
    if curl -f http://localhost:8002/health >/dev/null 2>&1; then
        log_success "TTS service is healthy"
    else
        log_error "TTS service health check failed"
    fi

    # Test STT service
    log_info "Testing STT service..."
    if curl -f http://localhost:8003/health >/dev/null 2>&1; then
        log_success "STT service is healthy"
    else
        log_error "STT service health check failed"
    fi

    log_success "Service tests completed"
}

# Watch mode
watch_tests() {
    log_info "Starting watch mode..."
    log_info "Tests will re-run when files change. Press Ctrl+C to stop."

    cd "$PROJECT_ROOT"

    if command -v pytest-watch >/dev/null 2>&1; then
        pytest-watch tests/
    else
        log_warning "pytest-watch not installed. Install with: pip install pytest-watch"
        log_info "Running tests once..."
        run_unit_tests
    fi
}

# Show coverage report
show_coverage() {
    if [[ "$COVERAGE" == true ]]; then
        log_info "Coverage report saved to htmlcov/index.html"
        log_info "Open in browser: file://$PROJECT_ROOT/htmlcov/index.html"

        # Try to open in browser
        if command -v xdg-open >/dev/null 2>&1; then
            xdg-open htmlcov/index.html 2>/dev/null || true
        elif command -v open >/dev/null 2>&1; then
            open htmlcov/index.html 2>/dev/null || true
        fi
    fi
}

# Main function
main() {
    echo ""
    log_info "================================"
    log_info "Morgan v2-0.0.1 Test Runner"
    log_info "================================"
    echo ""

    parse_args "$@"

    if [[ "$DOCKER" == true ]]; then
        run_docker_tests
        exit $?
    fi

    if [[ "$SERVICES" == true ]]; then
        test_services
        exit $?
    fi

    if [[ "$WATCH" == true ]]; then
        watch_tests
        exit $?
    fi

    check_venv
    setup_test_env

    local result=0

    case "$TEST_TYPE" in
        unit)
            run_unit_tests || result=$?
            ;;
        integration)
            run_integration_tests || result=$?
            ;;
        all)
            run_all_tests || result=$?
            ;;
    esac

    show_coverage

    echo ""
    if [[ $result -eq 0 ]]; then
        log_success "================================"
        log_success "All tests completed successfully!"
        log_success "================================"
    else
        log_error "================================"
        log_error "Some tests failed!"
        log_error "================================"
    fi
    echo ""

    exit $result
}

# Run main function
main "$@"
