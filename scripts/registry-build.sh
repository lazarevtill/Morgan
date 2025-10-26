#!/bin/bash

# Morgan AI Assistant - Private Registry Build Script
# This script builds and pushes images to the private Harbor registry

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
REGISTRY_BASE="harbor.in.lazarev.cloud/morgan"
REGISTRY_PROXY="harbor.in.lazarev.cloud/proxy"

# Service configurations
declare -A services=(
    ["core"]="core/Dockerfile"
    ["llm"]="services/llm/Dockerfile"
    ["tts"]="services/tts/Dockerfile"
    ["stt"]="services/stt/Dockerfile"
)

declare -A service_names=(
    ["core"]="core"
    ["llm"]="llm-service"
    ["tts"]="tts-service"
    ["stt"]="stt-service"
)

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

# Test Docker registry access
test_docker_registry_access() {
    print_header "Testing Docker Registry Access"

    echo -e "${BLUE}Testing connection to $REGISTRY_BASE...${NC}"

    if docker pull "$REGISTRY_PROXY/python:3.12-slim" &>/dev/null; then
        print_success "Registry access successful"
        docker rmi "$REGISTRY_PROXY/python:3.12-slim" &>/dev/null
        return 0
    else
        print_error "Cannot access private registry"
        echo -e "${YELLOW}Please ensure you are logged in to harbor.in.lazarev.cloud${NC}"
        echo -e "${GREEN}Run: docker login harbor.in.lazarev.cloud${NC}"
        return 1
    fi
}

# Build service image
build_service_image() {
    local service_name="$1"
    local tag="${2:-latest}"

    if [ ! -f "${services[$service_name]}" ]; then
        print_error "Unknown service: $service_name"
        return 1
    fi

    print_header "Building $service_name Service"

    local image_name="$REGISTRY_BASE/${service_names[$service_name]}:$tag"
    local local_image_name="morgan-${service_names[$service_name]}:$tag"

    echo -e "${BLUE}Building image: $image_name${NC}"
    echo -e "${BLUE}Also tagging as: $local_image_name${NC}"
    echo -e "${BLUE}Dockerfile: ${services[$service_name]}${NC}"
    echo -e "${BLUE}Context: .${NC}"

    if docker build -t "$image_name" -t "$local_image_name" -f "${services[$service_name]}" .; then
        print_success "Successfully built $image_name"
        print_success "Also tagged as $local_image_name"
        return 0
    else
        print_error "Build failed for $service_name"
        return 1
    fi
}

# Push service image
push_service_image() {
    local service_name="$1"
    local tag="${2:-latest}"

    print_header "Pushing $service_name Service"

    local image_name="$REGISTRY_BASE/${service_names[$service_name]}:$tag"

    echo -e "${BLUE}Pushing image: $image_name${NC}"

    if docker push "$image_name"; then
        print_success "Successfully pushed $image_name"
        return 0
    else
        print_error "Push failed for $image_name"
        return 1
    fi
}

# Pull service image
pull_service_image() {
    local service_name="$1"
    local tag="${2:-latest}"

    print_header "Pulling $service_name Service"

    local image_name="$REGISTRY_BASE/${service_names[$service_name]}:$tag"

    echo -e "${BLUE}Pulling image: $image_name${NC}"

    if docker pull "$image_name"; then
        print_success "Successfully pulled $image_name"
        return 0
    else
        print_warning "Pull failed for $image_name (image may not exist in registry)"
        return 1
    fi
}

# Build all services
build_all_services() {
    local tag="${1:-latest}"
    local success=true

    print_header "Building All Services"

    for service in "${!services[@]}"; do
        if ! build_service_image "$service" "$tag"; then
            print_warning "Stopping build due to failure in $service"
            success=false
            break
        fi
    done

    return $($success && echo 0 || echo 1)
}

# Push all services
push_all_services() {
    local tag="${1:-latest}"
    local success=true

    print_header "Pushing All Services"

    for service in "${!services[@]}"; do
        if ! push_service_image "$service" "$tag"; then
            print_warning "Stopping push due to failure in $service"
            success=false
            break
        fi
    done

    return $($success && echo 0 || echo 1)
}

# Pull all services
pull_all_services() {
    local tag="${1:-latest}"
    local success=true

    print_header "Pulling All Services"

    for service in "${!services[@]}"; do
        if ! pull_service_image "$service" "$tag"; then
            success=false
        fi
    done

    return $($success && echo 0 || echo 1)
}

# Show registry information
show_registry_info() {
    print_header "Private Registry Information"

    echo -e "${BLUE}Registry Base: $REGISTRY_BASE${NC}"
    echo -e "${BLUE}Registry Proxy: $REGISTRY_PROXY${NC}"
    echo ""
    echo -e "${BLUE}Service Images:${NC}"
    for service in "${!services[@]}"; do
        local image_name="$REGISTRY_BASE/${service_names[$service]}:latest"
        echo -e "${GREEN}  $service -> $image_name${NC}"
    done
    echo ""
    echo -e "${BLUE}Base Images Used:${NC}"
    echo -e "${YELLOW}  Python: $REGISTRY_PROXY/python:3.12-slim${NC}"
    echo -e "${YELLOW}  CUDA: $REGISTRY_PROXY/nvidia/cuda:13.0.1-devel-ubuntu22.04${NC}"
    echo -e "${YELLOW}  UV: ghcr.io/astral-sh/uv:latest${NC}"
    echo -e "${YELLOW}  Redis: $REGISTRY_PROXY/redis:7-alpine${NC}"
    echo -e "${YELLOW}  PostgreSQL: $REGISTRY_PROXY/postgres:17-alpine${NC}"
    echo ""
    echo -e "${BLUE}Commands:${NC}"
    echo -e "${GREEN}  docker login harbor.in.lazarev.cloud${NC}"
    echo -e "${GREEN}  ./scripts/registry-build.sh -b -p${NC}"
    echo -e "${GREEN}  docker-compose pull${NC}"
    echo -e "${GREEN}  docker-compose up -d${NC}"
}

# Show help
show_help() {
    print_header "Morgan AI Assistant - Private Registry Build Script"

    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -b, --build         Build images locally
    -p, --push          Push images to private registry
    -l, --pull          Pull images from private registry
    -c, --clean         Clean up local images after push
    -s, --service NAME  Build/push specific service (core, llm, tts, stt)
    -t, --tag TAG       Image tag (default: latest)
    -h, --help          Show this help message

Examples:
    $0 -b -p                 # Build and push all services
    $0 -s core -p            # Build and push core only
    $0 -l                    # Pull all images from registry
    $0 -c                    # Clean up after successful push

Registry Workflow:
    1. Login: docker login harbor.in.lazarev.cloud
    2. Build: $0 -b
    3. Push: $0 -p
    4. Deploy: docker-compose up -d

Available Services:
    - core     : Main orchestration service
    - llm      : OpenAI-compatible LLM client
    - tts      : Text-to-speech with CUDA
    - stt      : Speech-to-text with CUDA + integrated VAD

Image Naming Convention:
    harbor.in.lazarev.cloud/morgan/{service}:latest
    - harbor.in.lazarev.cloud/morgan/core:latest
    - harbor.in.lazarev.cloud/morgan/llm-service:latest
    - harbor.in.lazarev.cloud/morgan/tts-service:latest
    - harbor.in.lazarev.cloud/morgan/stt-service:latest

Base Images:
    - Python services: harbor.in.lazarev.cloud/proxy/python:3.12-slim
    - CUDA services: harbor.in.lazarev.cloud/proxy/nvidia/cuda:13.0.1-runtime-ubuntu22.04
    - UV: ghcr.io/astral-sh/uv:latest (public)

For troubleshooting, check:
    - Registry connectivity: docker login harbor.in.lazarev.cloud
    - Network access: curl https://harbor.in.lazarev.cloud
    - Permissions: Ensure you have push permissions to the morgan project
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
    local build=false
    local push=false
    local pull=false
    local clean=false
    local service=""
    local tag="latest"

    while [[ $# -gt 0 ]]; do
        case $1 in
            -b|--build)
                build=true
                shift
                ;;
            -p|--push)
                push=true
                shift
                ;;
            -l|--pull)
                pull=true
                shift
                ;;
            -c|--clean)
                clean=true
                shift
                ;;
            -s|--service)
                service="$2"
                shift 2
                ;;
            -t|--tag)
                tag="$2"
                shift 2
                ;;
            -h|--help)
                show_registry_info
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

    # Test registry access first
    if ! test_docker_registry_access; then
        print_error "Cannot access private registry. Please login first."
        echo -e "${BLUE}Run: docker login harbor.in.lazarev.cloud${NC}"
        exit 1
    fi

    local success=true

    if $build; then
        if [ -n "$service" ]; then
            success=$(build_service_image "$service" "$tag" && echo 0 || echo 1)
        else
            success=$(build_all_services "$tag" && echo 0 || echo 1)
        fi
    fi

    if $push; then
        if [ -n "$service" ]; then
            success=$(push_service_image "$service" "$tag" && echo 0 || echo 1)
        else
            success=$(push_all_services "$tag" && echo 0 || echo 1)
        fi
    fi

    if $pull; then
        if [ -n "$service" ]; then
            success=$(pull_service_image "$service" "$tag" && echo 0 || echo 1)
        else
            success=$(pull_all_services "$tag" && echo 0 || echo 1)
        fi
    fi

    if $clean; then
        print_header "Cleaning Up Local Images"
        for service in "${!services[@]}"; do
            local image_name="$REGISTRY_BASE/${service_names[$service]}:$tag"
            docker rmi "$image_name" &>/dev/null
            echo -e "${BLUE}Removed local image: $image_name${NC}"
        done
        print_success "Cleanup completed"
    fi

    # Show registry information if no specific action
    if ! $build && ! $push && ! $pull && ! $clean; then
        show_registry_info
    fi

    if $success; then
        print_success "Registry operation completed successfully!"
    else
        print_error "Some operations failed. Check the output above."
        exit 1
    fi
}

# Run main function
main "$@"
