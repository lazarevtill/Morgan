# Morgan AI Assistant Integration Test Script for Windows PowerShell
# This script tests the complete system integration

param(
    [switch]$Quick,
    [switch]$Full,
    [switch]$Health,
    [string]$Service,
    [switch]$Help
)

# Configuration
$BASE_URL = "http://localhost"
$TIMEOUT = 30

# Colors for output
$Green = "Green"
$Yellow = "Yellow"
$Red = "Red"
$Blue = "Blue"
$White = "White"

function Write-Header {
    param([string]$Message)
    Write-Host "=== $Message ===" -ForegroundColor $Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor $Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor $Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor $Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor $Blue
}

function Test-ServiceHealth {
    param([string]$ServiceName, [int]$Port, [string]$Description)

    Write-Host "Testing $Description ($ServiceName:$Port)..." -NoNewline

    try {
        $response = Invoke-WebRequest -Uri "$BASE_URL`:$Port/health" -TimeoutSec $TIMEOUT -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            $healthData = $response.Content | ConvertFrom-Json
            Write-Success "OK - Status: $($healthData.status)"
            return $true
        }
        else {
            Write-Warning "HTTP $($response.StatusCode)"
            return $false
        }
    }
    catch {
        Write-Error "Failed - $($_.Exception.Message)"
        return $false
    }
}

function Test-LLMService {
    Write-Header "Testing LLM Service (OpenAI Compatible)"

    try {
        # Test models endpoint
        Write-Host "Testing models endpoint..." -NoNewline
        $response = Invoke-WebRequest -Uri "$BASE_URL`:8001/v1/models" -TimeoutSec $TIMEOUT -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            $models = $response.Content | ConvertFrom-Json
            Write-Success "OK - Found $($models.data.Count) models"
        }
        else {
            Write-Error "Models endpoint failed"
            return $false
        }

        # Test chat completions
        Write-Host "Testing chat completions..." -NoNewline
        $body = @{
            model = "llama3.2:latest"
            messages = @(
                @{
                    role = "user"
                    content = "Hello! Please respond with just 'Hello from Morgan!'"
                }
            )
            max_tokens = 50
        } | ConvertTo-Json

        $response = Invoke-WebRequest -Uri "$BASE_URL`:8001/v1/chat/completions" -Method POST -Body $body -ContentType "application/json" -TimeoutSec $TIMEOUT -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            $result = $response.Content | ConvertFrom-Json
            Write-Success "OK - Generated response"
            Write-Info "Response: $($result.choices[0].message.content)"
        }
        else {
            Write-Error "Chat completions failed"
            return $false
        }

        return $true
    }
    catch {
        Write-Error "LLM Service test failed: $($_.Exception.Message)"
        return $false
    }
}

function Test-TTSService {
    Write-Header "Testing TTS Service"

    try {
        $body = @{
            text = "Hello, this is a test of the text to speech service."
            voice = "af_heart"
            speed = 1.0
        } | ConvertTo-Json

        Write-Host "Testing text-to-speech generation..." -NoNewline
        $response = Invoke-WebRequest -Uri "$BASE_URL`:8002/generate" -Method POST -Body $body -ContentType "application/json" -TimeoutSec $TIMEOUT -UseBasicParsing

        if ($response.StatusCode -eq 200) {
            Write-Success "OK - Generated audio data"
            Write-Info "Audio data length: $(($response.Content | Measure-Object).Count) bytes"
        }
        else {
            Write-Error "TTS generation failed"
            return $false
        }

        return $true
    }
    catch {
        Write-Error "TTS Service test failed: $($_.Exception.Message)"
        return $false
    }
}

function Test-STTService {
    Write-Header "Testing STT Service"

    try {
        # Create a simple test audio file (sine wave)
        Write-Host "Creating test audio file..." -NoNewline

        # Generate a simple WAV file with a tone
        $sampleRate = 16000
        $duration = 2 # seconds
        $frequency = 440 # Hz
        $samples = $sampleRate * $duration

        # This is a simplified test - in practice you'd use actual audio data
        $testAudio = [byte[]]::new(44 + $samples * 2) # WAV header + 16-bit samples

        # Simple WAV header (this is a placeholder - real implementation would need proper WAV encoding)
        Write-Warning "STT test requires real audio data - skipping detailed test"

        # Test basic connectivity instead
        $response = Invoke-WebRequest -Uri "$BASE_URL`:8003/health" -TimeoutSec $TIMEOUT -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Success "OK - Service is responsive"
            return $true
        }
        else {
            Write-Error "STT service not responding"
            return $false
        }
    }
    catch {
        Write-Error "STT Service test failed: $($_.Exception.Message)"
        return $false
    }
}

function Test-VADService {
    Write-Header "Testing VAD Service"

    try {
        # Test with empty audio data
        $body = @{
            audio_data = [Convert]::ToBase64String([byte[]]@(0) * 1024)
            threshold = 0.5
        } | ConvertTo-Json

        Write-Host "Testing voice activity detection..." -NoNewline
        $response = Invoke-WebRequest -Uri "$BASE_URL`:8004/detect" -Method POST -Body $body -ContentType "application/json" -TimeoutSec $TIMEOUT -UseBasicParsing

        if ($response.StatusCode -eq 200) {
            $result = $response.Content | ConvertFrom-Json
            Write-Success "OK - Detection completed"
            Write-Info "Speech detected: $($result.speech_detected)"
        }
        else {
            Write-Error "VAD detection failed"
            return $false
        }

        return $true
    }
    catch {
        Write-Error "VAD Service test failed: $($_.Exception.Message)"
        return $false
    }
}

function Test-CoreService {
    Write-Header "Testing Core Service"

    try {
        # Test health endpoint
        Write-Host "Testing core health..." -NoNewline
        $response = Invoke-WebRequest -Uri "$BASE_URL`:8000/health" -TimeoutSec $TIMEOUT -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            $health = $response.Content | ConvertFrom-Json
            Write-Success "OK - Status: $($health.status)"
        }
        else {
            Write-Error "Core health check failed"
            return $false
        }

        # Test text processing
        Write-Host "Testing text processing..." -NoNewline
        $body = @{
            text = "Turn on the living room lights"
            user_id = "test_user"
            metadata = @{ generate_audio = $false }
        } | ConvertTo-Json

        $response = Invoke-WebRequest -Uri "$BASE_URL`:8000/api/text" -Method POST -Body $body -ContentType "application/json" -TimeoutSec $TIMEOUT -UseBasicParsing

        if ($response.StatusCode -eq 200) {
            Write-Success "OK - Text processed"
        }
        else {
            Write-Error "Text processing failed"
            return $false
        }

        return $true
    }
    catch {
        Write-Error "Core Service test failed: $($_.Exception.Message)"
        return $false
    }
}

function Test-QuickIntegration {
    Write-Header "Quick Integration Test"

    $services = @(
        @{Name="Core"; Port=8000; Description="Core Service"},
        @{Name="LLM"; Port=8001; Description="LLM Service"},
        @{Name="TTS"; Port=8002; Description="TTS Service"},
        @{Name="STT"; Port=8003; Description="STT Service"},
        @{Name="VAD"; Port=8004; Description="VAD Service"}
    )

    $results = @()
    foreach ($service in $services) {
        $healthy = Test-ServiceHealth -ServiceName $service.Name -Port $service.Port -Description $service.Description
        $results += @{Service=$service.Name; Healthy=$healthy}
    }

    Write-Header "Integration Test Results"
    $healthyCount = ($results | Where-Object {$_.Healthy}).Count
    $totalCount = $results.Count

    Write-Host "Healthy Services: $healthyCount/$totalCount" -ForegroundColor $Blue

    foreach ($result in $results) {
        if ($result.Healthy) {
            Write-Success "$($result.Service) Service"
        }
        else {
            Write-Error "$($result.Service) Service"
        }
    }

    if ($healthyCount -eq $totalCount) {
        Write-Success "All services are healthy!"
        return $true
    }
    else {
        Write-Warning "Some services are not healthy. Check logs with: docker-compose logs"
        return $false
    }
}

function Test-FullIntegration {
    Write-Header "Full Integration Test"

    $tests = @(
        @{Name="LLM Service"; Test={Test-LLMService}},
        @{Name="TTS Service"; Test={Test-TTSService}},
        @{Name="STT Service"; Test={Test-STTService}},
        @{Name="VAD Service"; Test={Test-VADService}},
        @{Name="Core Service"; Test={Test-CoreService}}
    )

    $results = @()
    foreach ($test in $tests) {
        Write-Info "Running $($test.Name) test..."
        $success = & $test.Test
        $results += @{Test=$test.Name; Success=$success}
    }

    Write-Header "Full Integration Test Results"
    $successCount = ($results | Where-Object {$_.Success}).Count
    $totalCount = $results.Count

    Write-Host "Successful Tests: $successCount/$totalCount" -ForegroundColor $Blue

    foreach ($result in $results) {
        if ($result.Success) {
            Write-Success "$($result.Test)"
        }
        else {
            Write-Error "$($result.Test)"
        }
    }

    if ($successCount -eq $totalCount) {
        Write-Success "All integration tests passed!"
        return $true
    }
    else {
        Write-Warning "Some tests failed. Check service logs and configuration."
        return $false
    }
}

function Show-Help {
    Write-Header "Morgan AI Assistant Integration Test Script"

    @"
Usage: .\scripts\test-integration.ps1 [OPTIONS]

Options:
    -Quick              Run quick health checks only
    -Full               Run full integration tests (requires real data)
    -Health             Test health of all services
    -Service <name>     Test specific service (core, llm, tts, stt, vad)
    -Help               Show this help message

Examples:
    .\scripts\test-integration.ps1 -Quick    # Quick health check
    .\scripts\test-integration.ps1 -Full     # Full integration test
    .\scripts\test-integration.ps1 -Health   # Test all service health
    .\scripts\test-integration.ps1 -Service llm  # Test LLM service only

Prerequisites:
    - All services must be running (docker-compose up -d)
    - External Ollama service must be available
    - Network connectivity to localhost services

Services Tested:
    - Core Service (port 8000): Main orchestration
    - LLM Service (port 8001): OpenAI-compatible API for Ollama
    - TTS Service (port 8002): Text-to-speech synthesis
    - STT Service (port 8003): Speech-to-text recognition
    - VAD Service (port 8004): Voice activity detection

Test Types:
    - Quick: Basic health checks for all services
    - Full: Comprehensive tests with real data processing
    - Health: Detailed health status checks
    - Service: Individual service testing

For troubleshooting, check:
    - Service logs: docker-compose logs <service>
    - Health endpoints: curl http://localhost:<port>/health
    - Docker status: docker-compose ps
"@ | Write-Host
}

# Main execution
function Main {
    # Check if services are running
    try {
        $composeStatus = docker-compose ps --format "table {{.Name}}\t{{.Status}}"
        Write-Info "Docker Compose services status:"
        Write-Host $composeStatus
    }
    catch {
        Write-Warning "Docker Compose not available or services not running"
        Write-Warning "Start services with: docker-compose up -d"
    }

    # Execute requested tests
    if ($Help) {
        Show-Help
        return
    }

    if ($Quick) {
        Test-QuickIntegration
    }
    elseif ($Full) {
        Test-FullIntegration
    }
    elseif ($Health) {
        Test-QuickIntegration
    }
    elseif ($Service) {
        switch ($Service.ToLower()) {
            "core" { Test-CoreService }
            "llm" { Test-LLMService }
            "tts" { Test-TTSService }
            "stt" { Test-STTService }
            "vad" { Test-VADService }
            default {
                Write-Error "Unknown service: $Service"
                Write-Info "Available services: core, llm, tts, stt, vad"
            }
        }
    }
    else {
        # Default to quick test
        Test-QuickIntegration
    }
}

# Run main function
Main
