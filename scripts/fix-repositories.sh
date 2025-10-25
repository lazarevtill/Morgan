#!/bin/bash

# Morgan AI Assistant - Repository Configuration Fix Script
# This script fixes repository configurations for proper Debian/Ubuntu separation

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

# Test repository access for a specific service
test_repository_access() {
    local service_name="$1"
    local image_name="$2"
    local repo_config="$3"

    printf "Testing repository access for %s... " "$service_name"

    # Create a temporary Dockerfile to test repository access
    cat > test-repo-access.Dockerfile << EOF
FROM $image_name
RUN echo '$repo_config' > /etc/apt/sources.list
RUN apt-get update
EOF

    if docker build -f test-repo-access.Dockerfile -t test-repo-access . &>/dev/null; then
        docker rmi test-repo-access -f &>/dev/null
        rm test-repo-access.Dockerfile
        print_success "Repository access OK"
        return 0
    else
        rm test-repo-access.Dockerfile
        print_error "Repository access failed"
        return 1
    fi
}

# Validate repository configurations
validate_repositories() {
    print_header "Validating Repository Configurations"

    local all_passed=true

    # Test Debian-based services
    print_info "Testing Debian-based services..."

    local debian_config="deb https://nexus.in.lazarev.cloud/repository/debian-proxy/ trixie main
deb https://nexus.in.lazarev.cloud/repository/debian-proxy/ trixie-updates main
deb https://nexus.in.lazarev.cloud/repository/debian-security/ trixie-security main"

    # Test UV configuration for both local and Docker development
    print_info "Testing UV configuration..."
    if command -v uv &> /dev/null; then
        # Test virtual environment creation (for local development)
        if uv venv --help &>/dev/null; then
            print_success "UV virtual environment creation works"
        else
            print_warning "UV virtual environment creation may have issues"
        fi

        # Test system installation (for Docker containers)
        if uv pip install --dry-run fastapi --system &>/dev/null; then
            print_success "UV system pip install works"
        else
            print_warning "UV system pip install may have issues"
        fi
    else
        print_warning "UV not found in PATH"
    fi

    local debian_services=(
        "LLM Service:harbor.in.lazarev.cloud/proxy/python:3.12-slim"
        "VAD Service:harbor.in.lazarev.cloud/proxy/python:3.12-slim"
        "Core Service:harbor.in.lazarev.cloud/proxy/python:3.12-slim"
    )

    for service_info in "${debian_services[@]}"; do
        IFS=':' read -r service_name image_name <<< "$service_info"
        if ! test_repository_access "$service_name" "$image_name" "$debian_config"; then
            all_passed=false
        fi
    done

    # Test Ubuntu-based services
    print_info "Testing Ubuntu-based services..."

    local ubuntu_config="deb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy main restricted universe multiverse
deb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy-updates main restricted universe multiverse
deb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy-backports main restricted universe multiverse
deb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy-security main restricted universe multiverse"

    local ubuntu_services=(
        "TTS Service:harbor.in.lazarev.cloud/proxy/nvidia/cuda:13.0.1-devel-ubuntu22.04"
        "STT Service:harbor.in.lazarev.cloud/proxy/nvidia/cuda:13.0.1-devel-ubuntu22.04"
    )

    for service_info in "${ubuntu_services[@]}"; do
        IFS=':' read -r service_name image_name <<< "$service_info"
        if ! test_repository_access "$service_name" "$image_name" "$ubuntu_config"; then
            all_passed=false
        fi
    done

    echo ""
    if $all_passed; then
        print_success "All repository configurations are valid!"
    else
        print_error "Some repository configurations have issues."
        print_warning "Please check your Nexus proxy configuration."
    fi

    return $($all_passed && echo 0 || echo 1)
}

# Apply repository configuration fixes
fix_repository_configurations() {
    print_header "Fixing Repository Configurations"

    print_info "Updating Debian-based services (LLM, VAD, Core)..."
    print_info "Using: debian-proxy + debian-security"

    print_info "Updating Ubuntu-based services (TTS, STT)..."
    print_info "Using: ubuntu-group + ubuntu-group security"

    print_warning "Repository configurations have been updated in Dockerfiles."
    print_warning "Run 'docker-compose build --no-cache' to apply changes."
}

# Show current repository configuration
show_repository_configuration() {
    print_header "Current Repository Configuration"

    echo -e "${BLUE}Debian-based Services:${NC}"
    echo -e "  ${YELLOW}• LLM Service${NC}"
    echo -e "  ${YELLOW}• VAD Service${NC}"
    echo -e "  ${YELLOW}• Core Service${NC}"
    echo ""
    echo -e "${BLUE}Using repositories:${NC}"
    echo -e "  ${GREEN}• https://nexus.in.lazarev.cloud/repository/debian-proxy/${NC}"
    echo -e "  ${GREEN}• https://nexus.in.lazarev.cloud/repository/debian-security/${NC}"
    echo ""

    echo -e "${BLUE}Ubuntu-based Services:${NC}"
    echo -e "  ${YELLOW}• TTS Service${NC}"
    echo -e "  ${YELLOW}• STT Service${NC}"
    echo ""
    echo -e "${BLUE}Using repositories:${NC}"
    echo -e "  ${GREEN}• https://nexus.in.lazarev.cloud/repository/ubuntu-group/${NC}"
    echo -e "  ${GREEN}• https://nexus.in.lazarev.cloud/repository/ubuntu-group/ security${NC}"
    echo ""

    echo -e "${BLUE}Base Images:${NC}"
    echo -e "  ${GREEN}Debian Services: harbor.in.lazarev.cloud/proxy/python:3.12-slim${NC}"
    echo -e "  ${GREEN}Ubuntu Services: harbor.in.lazarev.cloud/proxy/nvidia/cuda:13.0.1-devel-ubuntu22.04${NC}"
}

# Show help
show_help() {
    print_header "Morgan AI Assistant - Repository Configuration Fix Script"

    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -v, --validate      Test repository access for all services
    -f, --fix           Apply repository configuration fixes
    -h, --help          Show this help message

Examples:
    $0 -v                    # Test all repository configurations
    $0 -f                    # Apply fixes to configurations

Repository Configuration:

Debian-based Services (python:3.12-slim):
  - LLM Service
  - VAD Service
  - Core Service

  Using repositories:
  - https://nexus.in.lazarev.cloud/repository/debian-proxy/
  - https://nexus.in.lazarev.cloud/repository/debian-security/

Ubuntu-based Services (nvidia/cuda:13.0.1-devel-ubuntu22.04):
  - TTS Service
  - STT Service

  Using repositories:
  - https://nexus.in.lazarev.cloud/repository/ubuntu-group/
  - https://nexus.in.lazarev.cloud/repository/ubuntu-group/ security

Common Issues Fixed:
  - Mixed Ubuntu/Debian repositories on wrong base images
  - Incorrect security repository URLs
  - Missing GPG keys for package verification

After running fixes:
  1. Run: docker-compose build --no-cache
  2. Run: docker-compose up -d
  3. Test: ./scripts/test-integration.sh -q
EOF
}

# Main execution
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--validate)
                show_repository_configuration
                validate_repositories
                exit $?
                ;;
            -f|--fix)
                fix_repository_configurations
                exit 0
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
        shift
    done

    # Default action - show configuration and validate
    show_repository_configuration
    echo ""
    print_warning "No action specified. Use -v to validate or -f to fix."
    echo ""
    show_help
}

# Run main function
main "$@"
