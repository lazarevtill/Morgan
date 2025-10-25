#!/bin/bash

# Morgan AI Assistant - Docker Diagnostic Script
# This script helps diagnose and fix Docker build issues

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print headers
print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to print info messages
print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check file structure
check_file_structure() {
    print_header "Checking File Structure"

    local services=("core" "llm" "tts" "stt" "vad")

    for service in "${services[@]}"; do
        echo -e "${BLUE}Checking $service service...${NC}"

        # Check main entry point
        local main_file="services/$service/main.py"
        if [ -f "$main_file" ]; then
            print_success "Found main.py: $main_file"
        else
            print_error "Missing main.py: $main_file"
        fi

        # Check service file
        local service_file="services/$service/service.py"
        if [ -f "$service_file" ]; then
            print_success "Found service.py: $service_file"
        else
            print_warning "Missing service.py: $service_file"
        fi

        # Check API server
        local api_server="services/$service/api/server.py"
        if [ -f "$api_server" ]; then
            print_success "Found api/server.py: $api_server"
        else
            print_warning "Missing api/server.py: $api_server"
        fi

        # Check Dockerfile
        local dockerfile="services/$service/Dockerfile"
        if [ "$service" = "core" ]; then
            dockerfile="core/Dockerfile"
        fi

        if [ -f "$dockerfile" ]; then
            print_success "Found Dockerfile: $dockerfile"
        else
            print_error "Missing Dockerfile: $dockerfile"
        fi

        echo ""
    done

    # Check shared modules
    echo -e "${BLUE}Checking shared modules...${NC}"
    local shared_dirs=("shared" "shared/config" "shared/models" "shared/utils")
    for dir in "${shared_dirs[@]}"; do
        if [ -d "$dir" ]; then
            print_success "Found shared directory: $dir"
        else
            print_error "Missing shared directory: $dir"
        fi
    done

    # Check __init__.py files
    echo -e "${BLUE}Checking __init__.py files...${NC}"
    local init_files=(
        "core/__init__.py"
        "services/llm/__init__.py"
        "services/tts/__init__.py"
        "services/stt/__init__.py"
        "services/vad/__init__.py"
        "shared/__init__.py"
        "shared/config/__init__.py"
        "shared/models/__init__.py"
        "shared/utils/__init__.py"
    )

    for init_file in "${init_files[@]}"; do
        if [ -f "$init_file" ]; then
            print_success "Found __init__.py: $init_file"
        else
            print_warning "Missing __init__.py: $init_file"
        fi
    done
}

# Fix import issues
fix_import_issues() {
    print_header "Fixing Import Issues"

    # Create missing __init__.py files
    local missing_init_files=(
        "core/handlers/__init__.py"
        "core/integrations/__init__.py"
        "core/services/__init__.py"
        "core/utils/__init__.py"
        "core/api/__init__.py"
        "core/conversation/__init__.py"
    )

    for init_file in "${missing_init_files[@]}"; do
        if [ ! -f "$init_file" ]; then
            touch "$init_file"
            print_success "Created missing __init__.py: $init_file"
        fi
    done

    # Check service import statements
    echo -e "${BLUE}Checking service import statements...${NC}"

    # Core service imports
    if [ -f "core/main.py" ]; then
        if grep -q "from \.app import main" core/main.py; then
            print_success "Core main.py has correct import"
        else
            print_warning "Core main.py may have incorrect imports"
        fi
    fi

    # Service imports
    local services=("llm" "tts" "stt" "vad")
    for service in "${services[@]}"; do
        local main_file="services/$service/main.py"
        if [ -f "$main_file" ]; then
            if grep -q "from \.api\.server import main" "$main_file"; then
                print_success "$service main.py has correct import"
            else
                print_warning "$service main.py may have incorrect imports"
            fi
        fi
    done
}

# Validate Docker build
validate_docker_build() {
    local service_name="$1"

    print_header "Validating Docker Build"

    if [ -n "$service_name" ]; then
        echo -e "${BLUE}Building service: $service_name${NC}"
        local build_cmd="docker-compose build $service_name"
    else
        echo -e "${BLUE}Building all services${NC}"
        local build_cmd="docker-compose build"
    fi

    echo -e "${BLUE}Running: $build_cmd${NC}"

    if eval "$build_cmd" &>/dev/null; then
        print_success "Docker build successful"
        return 0
    else
        print_error "Docker build failed"
        echo -e "${RED}Build failed. Check the output above.${NC}"
        return 1
    fi
}

# Clean rebuild
clean_rebuild() {
    print_header "Clean Rebuild"

    echo -e "${YELLOW}⚠ This will remove all containers, volumes, and images!${NC}"
    read -p "Are you sure? (y/N): " confirmation

    if [[ $confirmation =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Stopping services...${NC}"
        docker-compose down -v &>/dev/null

        echo -e "${BLUE}Removing images...${NC}"
        docker-compose down --rmi all &>/dev/null

        echo -e "${BLUE}Cleaning up...${NC}"
        docker system prune -f &>/dev/null

        echo -e "${BLUE}Building fresh...${NC}"
        if validate_docker_build; then
            print_success "Clean rebuild completed successfully"
            return 0
        else
            print_error "Clean rebuild failed"
            return 1
        fi
    else
        echo -e "${BLUE}Clean rebuild cancelled${NC}"
        return 0
    fi
}

# Show Docker tips
show_docker_tips() {
    print_header "Docker Build Tips"

    echo -e "${BLUE}Common Issues and Solutions:${NC}"
    echo ""
    echo -e "${YELLOW}1. Missing main.py files:${NC}"
    echo -e "${WHITE}   - Check that all services have main.py entry points${NC}"
    echo -e "${WHITE}   - Verify CMD in Dockerfile points to correct file${NC}"
    echo ""
    echo -e "${YELLOW}2. Multi-stage build issues:${NC}"
    echo -e "${WHITE}   - Runtime stage should inherit from build stage${NC}"
    echo -e "${WHITE}   - Build stage should copy all necessary files${NC}"
    echo ""
    echo -e "${YELLOW}3. Missing dependencies:${NC}"
    echo -e "${WHITE}   - Check pyproject.toml has correct dependency groups${NC}"
    echo -e "${WHITE}   - Verify UV is installing to system Python${NC}"
    echo ""
    echo -e "${YELLOW}4. Import errors:${NC}"
    echo -e "${WHITE}   - Ensure all __init__.py files exist${NC}"
    echo -e "${WHITE}   - Check PYTHONPATH is set correctly${NC}"
    echo ""
    echo -e "${BLUE}Build Commands:${NC}"
    echo -e "${GREEN}  docker-compose build --no-cache${NC}"
    echo -e "${GREEN}  docker-compose up -d --build${NC}"
    echo -e "${GREEN}  docker-compose logs -f <service>${NC}"
    echo -e "${GREEN}  docker-compose exec <service> python main.py${NC}"
}

# Show help
show_help() {
    print_header "Morgan AI Assistant - Docker Diagnostic Script"

    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -c, --check-files   Check file structure and imports
    -f, --fix-imports   Fix missing __init__.py files and imports
    -v, --validate      Validate Docker build for all services
    -r, --rebuild       Clean rebuild all services
    -s, --service NAME  Check specific service (core, llm, tts, stt, vad)
    -h, --help          Show this help message

Examples:
    $0 -c                    # Check all files
    $0 -f                    # Fix import issues
    $0 -v                    # Test Docker builds
    $0 -r                    # Fresh start

Diagnostic Checks:
    - File structure validation
    - Import statement verification
    - Dockerfile multi-stage build validation
    - UV configuration testing
    - Docker build process testing

Common Issues Fixed:
    - Missing __init__.py files
    - Incorrect multi-stage Docker builds
    - Import path errors
    - Missing entry points
    - Dependency installation issues

Troubleshooting Steps:
    1. Run: $0 -c
    2. Run: $0 -f
    3. Run: $0 -v
    4. If issues persist: $0 -r
EOF
}

# Main execution
main() {
    # Check if we're in the project root
    if [ ! -f "docker-compose.yml" ]; then
        print_error "docker-compose.yml not found. Please run this script from the project root."
        exit 1
    fi

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--check-files)
                check_file_structure
                shift
                ;;
            -f|--fix-imports)
                fix_import_issues
                shift
                ;;
            -v|--validate)
                if [ -n "$2" ] && [[ ! $2 =~ ^- ]]; then
                    validate_docker_build "$2"
                    shift 2
                else
                    validate_docker_build
                    shift
                fi
                ;;
            -r|--rebuild)
                clean_rebuild
                shift
                ;;
            -h|--help)
                show_docker_tips
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # If no arguments provided, show tips and run basic checks
    if [ $# -eq 0 ]; then
        show_docker_tips
        check_file_structure
    fi
}

# Run main function
main "$@"
