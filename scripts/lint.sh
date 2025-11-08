#!/usr/bin/env bash
#
# Morgan v2-0.0.1 - Comprehensive Linting Script
#
# This script runs all linters and code quality checks.
# It is idempotent and can be run multiple times safely.
#
# Usage:
#   ./scripts/lint.sh [OPTIONS]
#
# Options:
#   --fix              Auto-fix issues where possible
#   --check            Check only (no fixes) - default
#   --format           Format code with Black and isort
#   --type             Run type checking with mypy
#   --security         Run security checks
#   --complexity       Check code complexity
#   --all              Run all checks
#   --fast             Skip slow checks
#   --ci               CI mode (fail on errors)
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
FIX=false
CHECK=true
FORMAT=false
TYPE_CHECK=false
SECURITY=false
COMPLEXITY=false
RUN_ALL=false
FAST=false
CI_MODE=false

# Exit code tracking
EXIT_CODE=0

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
            --fix)
                FIX=true
                CHECK=false
                shift
                ;;
            --check)
                CHECK=true
                FIX=false
                shift
                ;;
            --format)
                FORMAT=true
                shift
                ;;
            --type)
                TYPE_CHECK=true
                shift
                ;;
            --security)
                SECURITY=true
                shift
                ;;
            --complexity)
                COMPLEXITY=true
                shift
                ;;
            --all)
                RUN_ALL=true
                shift
                ;;
            --fast)
                FAST=true
                shift
                ;;
            --ci)
                CI_MODE=true
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

    # If --all is specified, enable all checks
    if [[ "$RUN_ALL" == true ]]; then
        FORMAT=true
        TYPE_CHECK=true
        SECURITY=true
        COMPLEXITY=true
    fi
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

# Update exit code
update_exit_code() {
    local result=$1
    if [[ $result -ne 0 ]]; then
        EXIT_CODE=1
    fi
}

# Run Black formatter
run_black() {
    log_info "Running Black (Python formatter)..."

    cd "$PROJECT_ROOT"

    local black_args="core/ services/ shared/ tests/"

    if [[ "$FIX" == true ]]; then
        if black $black_args; then
            log_success "Black formatting completed"
        else
            log_error "Black formatting failed"
            update_exit_code 1
        fi
    else
        if black --check --diff $black_args; then
            log_success "Black check passed"
        else
            log_warning "Black formatting needed. Run with --fix to auto-format."
            update_exit_code 1
        fi
    fi
}

# Run isort
run_isort() {
    log_info "Running isort (import sorter)..."

    cd "$PROJECT_ROOT"

    local isort_args="core/ services/ shared/ tests/ --profile black"

    if [[ "$FIX" == true ]]; then
        if isort $isort_args; then
            log_success "isort completed"
        else
            log_error "isort failed"
            update_exit_code 1
        fi
    else
        if isort --check-only --diff $isort_args; then
            log_success "isort check passed"
        else
            log_warning "Import sorting needed. Run with --fix to auto-sort."
            update_exit_code 1
        fi
    fi
}

# Run Ruff linter
run_ruff() {
    log_info "Running Ruff (fast Python linter)..."

    cd "$PROJECT_ROOT"

    local ruff_args="check core/ services/ shared/ tests/"

    if [[ "$FIX" == true ]]; then
        ruff_args="$ruff_args --fix"
    fi

    if ruff $ruff_args; then
        log_success "Ruff check passed"
    else
        log_warning "Ruff found issues"
        update_exit_code 1
    fi
}

# Run mypy type checker
run_mypy() {
    if [[ "$TYPE_CHECK" == false ]] && [[ "$RUN_ALL" == false ]]; then
        return
    fi

    log_info "Running mypy (type checker)..."

    cd "$PROJECT_ROOT"

    if mypy core/ services/ shared/ --ignore-missing-imports --no-strict-optional; then
        log_success "mypy check passed"
    else
        log_warning "mypy found type issues"
        update_exit_code 1
    fi
}

# Run Bandit security scanner
run_bandit() {
    if [[ "$SECURITY" == false ]] && [[ "$RUN_ALL" == false ]]; then
        return
    fi

    log_info "Running Bandit (security scanner)..."

    cd "$PROJECT_ROOT"

    if bandit -r core/ services/ shared/ -ll; then
        log_success "Bandit security check passed"
    else
        log_warning "Bandit found security issues"
        update_exit_code 1
    fi
}

# Run safety check
run_safety() {
    if [[ "$SECURITY" == false ]] && [[ "$RUN_ALL" == false ]]; then
        return
    fi

    log_info "Running Safety (dependency security checker)..."

    cd "$PROJECT_ROOT"

    if safety check --json || true; then
        log_success "Safety check completed"
    else
        log_warning "Safety found vulnerabilities"
        # Don't fail on safety issues in non-CI mode
        if [[ "$CI_MODE" == true ]]; then
            update_exit_code 1
        fi
    fi
}

# Check code complexity
check_complexity() {
    if [[ "$COMPLEXITY" == false ]] && [[ "$RUN_ALL" == false ]]; then
        return
    fi

    log_info "Checking code complexity..."

    cd "$PROJECT_ROOT"

    if command -v radon >/dev/null 2>&1; then
        log_info "Cyclomatic complexity:"
        radon cc core/ services/ shared/ -a -nb

        log_info "Maintainability index:"
        radon mi core/ services/ shared/ -nb

        log_success "Complexity check completed"
    else
        log_warning "radon not installed. Install with: pip install radon"
    fi
}

# Run YAML linter
run_yamllint() {
    if [[ "$FAST" == true ]]; then
        return
    fi

    log_info "Running yamllint..."

    cd "$PROJECT_ROOT"

    if command -v yamllint >/dev/null 2>&1; then
        if yamllint -c .yamllint.yml config/ .github/workflows/ docker-compose.yml; then
            log_success "YAML lint passed"
        else
            log_warning "YAML lint found issues"
            update_exit_code 1
        fi
    else
        log_warning "yamllint not installed. Install with: pip install yamllint"
    fi
}

# Run Dockerfile linter
run_hadolint() {
    if [[ "$FAST" == true ]]; then
        return
    fi

    log_info "Running hadolint (Dockerfile linter)..."

    cd "$PROJECT_ROOT"

    if command -v hadolint >/dev/null 2>&1; then
        local dockerfiles=(
            "core/Dockerfile"
            "services/llm/Dockerfile"
            "services/tts/Dockerfile"
            "services/stt/Dockerfile"
        )

        for dockerfile in "${dockerfiles[@]}"; do
            if [[ -f "$dockerfile" ]]; then
                log_info "Checking $dockerfile..."
                if hadolint "$dockerfile" --ignore DL3008 --ignore DL3013 --ignore DL3015; then
                    log_success "$dockerfile passed"
                else
                    log_warning "$dockerfile has issues"
                    update_exit_code 1
                fi
            fi
        done
    else
        log_warning "hadolint not installed. See: https://github.com/hadolint/hadolint"
    fi
}

# Run ShellCheck
run_shellcheck() {
    if [[ "$FAST" == true ]]; then
        return
    fi

    log_info "Running shellcheck (shell script linter)..."

    cd "$PROJECT_ROOT"

    if command -v shellcheck >/dev/null 2>&1; then
        if find scripts/ -name "*.sh" -type f -exec shellcheck {} +; then
            log_success "shellcheck passed"
        else
            log_warning "shellcheck found issues"
            update_exit_code 1
        fi
    else
        log_warning "shellcheck not installed. Install with your package manager."
    fi
}

# Check Python syntax
check_python_syntax() {
    log_info "Checking Python syntax..."

    cd "$PROJECT_ROOT"

    local syntax_errors=0

    while IFS= read -r -d '' file; do
        if ! python3 -m py_compile "$file" 2>/dev/null; then
            log_error "Syntax error in $file"
            syntax_errors=$((syntax_errors + 1))
        fi
    done < <(find core/ services/ shared/ tests/ -name "*.py" -print0)

    if [[ $syntax_errors -eq 0 ]]; then
        log_success "Python syntax check passed"
    else
        log_error "Found $syntax_errors syntax errors"
        update_exit_code 1
    fi
}

# Show summary
show_summary() {
    echo ""
    log_info "================================"
    log_info "Linting Summary"
    log_info "================================"
    echo ""

    if [[ $EXIT_CODE -eq 0 ]]; then
        log_success "All checks passed!"
    else
        log_warning "Some checks found issues"
        if [[ "$FIX" == false ]]; then
            log_info "Run with --fix to auto-fix formatting issues"
        fi
    fi

    echo ""
}

# Main function
main() {
    echo ""
    log_info "================================"
    log_info "Morgan v2-0.0.1 Linter"
    log_info "================================"
    echo ""

    parse_args "$@"
    check_venv

    cd "$PROJECT_ROOT"

    # Always run these
    check_python_syntax

    # Format checks
    if [[ "$FORMAT" == true ]] || [[ "$RUN_ALL" == true ]] || [[ "$CHECK" == true ]]; then
        run_black
        run_isort
    fi

    # Always run Ruff
    run_ruff

    # Optional checks
    run_mypy
    run_bandit
    run_safety
    check_complexity

    # Additional linters
    if [[ "$FAST" == false ]]; then
        run_yamllint
        run_hadolint
        run_shellcheck
    fi

    show_summary

    if [[ "$CI_MODE" == true ]]; then
        exit $EXIT_CODE
    fi

    exit 0
}

# Run main function
main "$@"
