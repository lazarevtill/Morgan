#!/bin/bash

# Morgan AI Assistant Integration Test Script
# This script tests the complete system integration

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

BASE_URL="http://localhost"
TIMEOUT=30

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

# Test service health
test_service_health() {
    local service_name=$1
    local port=$2
    local description=$3

    printf "Testing %s (%s:%d)... " "$description" "$service_name" "$port"

    if timeout "$TIMEOUT" curl -f "$BASE_URL:$port/health" &>/dev/null; then
        print_success "OK"
        return 0
    else
        print_error "Failed"
        return 1
    fi
}

# Test LLM service
test_llm_service() {
    print_header "Testing LLM Service (OpenAI Compatible)"

    # Test models endpoint
    print_info "Testing models endpoint..."
    if ! timeout "$TIMEOUT" curl -f "$BASE_URL:8001/v1/models" &>/dev/null; then
        print_error "Models endpoint failed"
        return 1
    fi
    print_success "Models endpoint OK"

    # Test chat completions
    print_info "Testing chat completions..."
    local response=$(timeout "$TIMEOUT" curl -s -X POST "$BASE_URL:8001/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "llama3.2:latest",
            "messages": [
                {"role": "user", "content": "Hello! Please respond with just \"Hello from Morgan!\""}
            ],
            "max_tokens": 50
        }')

    if [ $? -eq 0 ] && echo "$response" | grep -q "Hello from Morgan"; then
        print_success "Chat completions OK"
        print_info "Response: $(echo "$response" | grep -o '"content":"[^"]*"' | cut -d'"' -f4)"
        return 0
    else
        print_error "Chat completions failed"
        return 1
    fi
}

# Test TTS service
test_tts_service() {
    print_header "Testing TTS Service"

    print_info "Testing text-to-speech generation..."
    local response=$(timeout "$TIMEOUT" curl -s -X POST "$BASE_URL:8002/generate" \
        -H "Content-Type: application/json" \
        -d '{
            "text": "Hello, this is a test of the text to speech service.",
            "voice": "af_heart",
            "speed": 1.0
        }')

    if [ $? -eq 0 ]; then
        print_success "TTS generation OK"
        print_info "Audio data length: $(echo "$response" | wc -c) bytes"
        return 0
    else
        print_error "TTS generation failed"
        return 1
    fi
}

# Test STT service
test_stt_service() {
    print_header "Testing STT Service"

    print_warning "STT test requires real audio data - performing basic connectivity test"

    if timeout "$TIMEOUT" curl -f "$BASE_URL:8003/health" &>/dev/null; then
        print_success "STT service is responsive"
        return 0
    else
        print_error "STT service not responding"
        return 1
    fi
}

# Test VAD service (integrated with STT)
test_vad_service() {
    print_header "Testing VAD Service (Integrated with STT)"

    print_info "Testing VAD through STT service..."
    if response=$(timeout "$TIMEOUT" curl -s "$BASE_URL:8003/health"); then
        local vad_enabled=$(echo "$response" | grep -o '"vad_enabled":[^,]*' | cut -d':' -f2 | tr -d '"')
        local device=$(echo "$response" | grep -o '"device":[^,]*' | cut -d':' -f2 | tr -d '"')

        if [ "$vad_enabled" = "true" ]; then
            print_success "VAD detection OK"
            print_info "VAD enabled: $vad_enabled, Device: $device"
            return 0
        else
            print_warning "VAD not enabled in STT service"
            return 1
        fi
    else
        print_error "STT service health check failed"
        return 1
    fi
}

# Test Core service
test_core_service() {
    print_header "Testing Core Service"

    # Test health endpoint
    print_info "Testing core health..."
    if ! timeout "$TIMEOUT" curl -f "$BASE_URL:8000/health" &>/dev/null; then
        print_error "Core health check failed"
        return 1
    fi
    print_success "Core health OK"

    # Test text processing
    print_info "Testing text processing..."
    local response=$(timeout "$TIMEOUT" curl -s -X POST "$BASE_URL:8000/api/text" \
        -H "Content-Type: application/json" \
        -d '{
            "text": "Turn on the living room lights",
            "user_id": "test_user",
            "metadata": {"generate_audio": false}
        }')

    if [ $? -eq 0 ]; then
        print_success "Text processing OK"
        return 0
    else
        print_error "Text processing failed"
        return 1
    fi
}

# Quick integration test
test_quick_integration() {
    print_header "Quick Integration Test"

    local services=(
        "core:8000:Core Service"
        "llm:8001:LLM Service"
        "tts:8002:TTS Service"
        "stt:8003:STT Service (with VAD)"
    )

    local results=()
    local total=0
    local healthy=0

    for service in "${services[@]}"; do
        IFS=':' read -r name port description <<< "$service"
        total=$((total + 1))

        if test_service_health "$name" "$port" "$description"; then
            healthy=$((healthy + 1))
            results+=("$name:healthy")
        else
            results+=("$name:unhealthy")
        fi
    done

    print_header "Integration Test Results"
    echo -e "${BLUE}Healthy Services: $healthy/$total${NC}"

    for result in "${results[@]}"; do
        IFS=':' read -r name status <<< "$result"
        if [ "$status" = "healthy" ]; then
            print_success "$name Service"
        else
            print_error "$name Service"
        fi
    done

    if [ "$healthy" -eq "$total" ]; then
        print_success "All services are healthy!"
        return 0
    else
        print_warning "Some services are not healthy. Check logs with: docker-compose logs"
        return 1
    fi
}

# Full integration test
test_full_integration() {
    print_header "Full Integration Test"

    local tests=(
        "test_llm_service:LLM Service"
        "test_tts_service:TTS Service"
        "test_stt_service:STT Service"
        "test_vad_service:STT+VAD Service"
        "test_core_service:Core Service"
    )

    local results=()
    local total=0
    local successful=0

    for test in "${tests[@]}"; do
        IFS=':' read -r test_func description <<< "$test"
        total=$((total + 1))

        print_info "Running $description test..."
        if $test_func; then
            successful=$((successful + 1))
            results+=("$description:success")
        else
            results+=("$description:failed")
        fi
    done

    print_header "Full Integration Test Results"
    echo -e "${BLUE}Successful Tests: $successful/$total${NC}"

    for result in "${results[@]}"; do
        IFS=':' read -r description status <<< "$result"
        if [ "$status" = "success" ]; then
            print_success "$description"
        else
            print_error "$description"
        fi
    done

    if [ "$successful" -eq "$total" ]; then
        print_success "All integration tests passed!"
        return 0
    else
        print_warning "Some tests failed. Check service logs and configuration."
        return 1
    fi
}

# Show help
show_help() {
    print_header "Morgan AI Assistant Integration Test Script"

    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -q, --quick         Run quick health checks only
    -f, --full          Run full integration tests (requires real data)
    -h, --health        Test health of all services
    -s, --service NAME  Test specific service (core, llm, tts, stt, vad)
    --help              Show this help message

Examples:
    $0 -q                    # Quick health check
    $0 -f                    # Full integration test
    $0 -h                    # Test all service health
    $0 -s llm                # Test LLM service only

Prerequisites:
    - All services must be running (docker-compose up -d)
    - External Ollama service must be available
    - Network connectivity to localhost services

Services Tested:
    - Core Service (port 8000): Main orchestration and API gateway
    - LLM Service (port 8001): OpenAI-compatible API for external Ollama
    - TTS Service (port 8002): Text-to-speech synthesis with CUDA support
    - STT Service (port 8003): Speech-to-text recognition with integrated VAD
    - VAD Service: Voice activity detection (integrated with STT)

Test Types:
    - Quick: Basic health checks for all services
    - Full: Comprehensive tests with real data processing
    - Health: Detailed health status checks
    - Service: Individual service testing

For troubleshooting, check:
    - Service logs: docker-compose logs <service>
    - Health endpoints: curl http://localhost:<port>/health
    - Docker status: docker-compose ps
EOF
}

# Main execution
main() {
    # Check if services are running
    if command -v docker-compose &> /dev/null; then
        print_info "Docker Compose services status:"
        docker-compose ps --format "table {{.Name}}\t{{.Status}}" || print_warning "Docker Compose not available or services not running"
    else
        print_warning "Docker Compose not available"
    fi

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -q|--quick)
                test_quick_integration
                exit $?
                ;;
            -f|--full)
                test_full_integration
                exit $?
                ;;
            -h|--health)
                test_quick_integration
                exit $?
                ;;
            -s|--service)
                shift
                service="$1"
                case $service in
                    core) test_core_service ;;
                    llm) test_llm_service ;;
                    tts) test_tts_service ;;
                    stt) test_stt_service ;;
                    vad) test_vad_service ;;
                    *)
                        print_error "Unknown service: $service"
                        print_info "Available services: core, llm, tts, stt, vad (integrated with STT)"
                        exit 1
                        ;;
                esac
                exit $?
                ;;
            --help)
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

    # Default to quick test
    test_quick_integration
    exit $?
}

# Run main function
main "$@"
