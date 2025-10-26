#!/bin/bash

# Morgan AI Assistant - Service Testing Script
# Tests all service endpoints and functionality

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_test() {
    echo -e "${BLUE}Testing: $1${NC}"
}

print_pass() {
    echo -e "${GREEN}✓ PASS: $1${NC}"
}

print_fail() {
    echo -e "${RED}✗ FAIL: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ WARNING: $1${NC}"
}

echo "=== Morgan AI Assistant - Service Tests ==="
echo ""

# Test Core Service
print_test "Core Service Health"
CORE_HEALTH=$(curl -s http://localhost:8000/health)
if echo "$CORE_HEALTH" | grep -q '"status"'; then
    print_pass "Core service is responding"

    STATUS=$(echo "$CORE_HEALTH" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    if [ "$STATUS" = "healthy" ]; then
        print_pass "Core service status: healthy"
    elif [ "$STATUS" = "degraded" ]; then
        print_warning "Core service status: degraded"
    else
        print_fail "Core service status: $STATUS"
    fi
else
    print_fail "Core service not responding"
    exit 1
fi

# Test LLM Service
print_test "LLM Service Health"
LLM_HEALTH=$(curl -s http://localhost:8001/health)
if echo "$LLM_HEALTH" | grep -q '"status"'; then
    print_pass "LLM service is responding"
else
    print_fail "LLM service not responding"
fi

# Test TTS Service
print_test "TTS Service Health"
TTS_HEALTH=$(curl -s http://localhost:8002/health)
if echo "$TTS_HEALTH" | grep -q '"status"'; then
    print_pass "TTS service is responding"
else
    print_fail "TTS service not responding"
fi

# Test STT Service
print_test "STT Service Health"
STT_HEALTH=$(curl -s http://localhost:8003/health)
if echo "$STT_HEALTH" | grep -q '"status"'; then
    print_pass "STT service is responding"
else
    print_fail "STT service not responding"
fi

# Test Core Text Processing
print_test "Core Text Processing"
TEXT_RESPONSE=$(curl -s -X POST http://localhost:8000/api/text \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello","user_id":"test"}')
if echo "$TEXT_RESPONSE" | grep -q '"text"'; then
    print_pass "Text processing working"
else
    print_fail "Text processing failed"
fi

# Test Request ID Middleware
print_test "Request ID Middleware"
HEADERS=$(curl -s -v http://localhost:8000/health 2>&1)
if echo "$HEADERS" | grep -q "X-Request-ID"; then
    print_pass "Request ID middleware working"
else
    print_fail "Request ID middleware not found"
fi

# Test Timing Middleware
print_test "Timing Middleware"
if echo "$HEADERS" | grep -q "X-Process-Time"; then
    print_pass "Timing middleware working"
else
    print_fail "Timing middleware not found"
fi

# Check Service Configuration
print_test "Service Configuration"
echo "$CORE_HEALTH" | grep -q '"llm": true' && print_pass "LLM service registered" || print_fail "LLM service not registered"
echo "$CORE_HEALTH" | grep -q '"tts": true' && print_pass "TTS service registered" || print_fail "TTS service not registered"
echo "$CORE_HEALTH" | grep -q '"stt": true' && print_pass "STT service registered" || print_fail "STT service not registered"

# Check VAD is removed
print_test "VAD Service Removal"
if ! echo "$CORE_HEALTH" | grep -q '"vad"'; then
    print_pass "VAD service successfully removed"
else
    print_fail "VAD service still referenced"
fi

echo ""
echo "=== Test Summary ==="
echo ""
echo -e "${BLUE}Services Status:${NC}"
docker-compose ps

echo ""
echo -e "${BLUE}Service URLs:${NC}"
echo -e "  Core:   http://localhost:8000"
echo -e "  LLM:    http://localhost:8001"
echo -e "  TTS:    http://localhost:8002"
echo -e "  STT:    http://localhost:8003"
echo ""
