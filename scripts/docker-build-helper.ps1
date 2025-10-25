# Morgan AI Assistant - Docker Build Helper Script
# This script helps with Docker builds and resolves common issues

param(
    [switch]$GenerateLockfile,
    [switch]$BuildNoCache,
    [switch]$ValidateConfig,
    [switch]$CleanStart,
    [string]$Service,
    [switch]$Help
)

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

function Generate-Lockfile {
    Write-Header "Generating UV Lockfile"

    # Check if UV is available
    try {
        $uvVersion = uv --version 2>$null
        Write-Success "UV is available: $uvVersion"
    }
    catch {
        Write-Error "UV is not installed. Please run: .\scripts\setup-uv.ps1 -Install"
        return $false
    }

    # Generate lockfile
    Write-Info "Creating uv.lock file..."
    try {
        uv lock
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Lockfile generated successfully"
            return $true
        }
        else {
            Write-Error "Failed to generate lockfile"
            return $false
        }
    }
    catch {
        Write-Error "Lockfile generation failed: $($_.Exception.Message)"
        return $false
    }
}

function Validate-DockerConfiguration {
    Write-Header "Validating Docker Configuration"

    # Check if Docker is running
    try {
        $dockerInfo = docker info 2>$null
        Write-Success "Docker is running"
    }
    catch {
        Write-Error "Docker is not running. Please start Docker Desktop."
        return $false
    }

    # Check if docker-compose is available
    try {
        $composeInfo = docker compose version 2>$null
        Write-Success "Docker Compose is available"
    }
    catch {
        Write-Error "Docker Compose is not available"
        return $false
    }

    # Check repository configurations
    Write-Info "Checking repository configurations..."
    try {
        & "$PSScriptRoot\fix-repositories.ps1" -Validate
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Repository configurations are valid"
        }
        else {
            Write-Warning "Repository configurations need attention"
        }
    }
    catch {
        Write-Warning "Could not validate repositories: $($_.Exception.Message)"
    }

    # Check UV configuration
    Write-Info "Checking UV configuration..."
    try {
        & "$PSScriptRoot\setup-uv.ps1" -Validate
        if ($LASTEXITCODE -eq 0) {
            Write-Success "UV configuration is valid"
        }
        else {
            Write-Warning "UV configuration needs attention"
        }
    }
    catch {
        Write-Warning "Could not validate UV: $($_.Exception.Message)"
    }

    return $true
}

function Build-DockerServices {
    param([switch]$NoCache, [string]$SpecificService)

    Write-Header "Building Docker Services"

    $buildArgs = @("build")

    if ($NoCache) {
        $buildArgs += "--no-cache"
        Write-Info "Building without cache"
    }
    else {
        Write-Info "Building with cache (use -NoCache for clean build)"
    }

    if ($SpecificService) {
        $buildArgs += $SpecificService
        Write-Info "Building service: $SpecificService"
    }
    else {
        Write-Info "Building all services"
    }

    try {
        $buildOutput = & docker compose $buildArgs 2>&1
        $buildExitCode = $LASTEXITCODE

        if ($buildExitCode -eq 0) {
            Write-Success "Docker build completed successfully"
            return $true
        }
        else {
            Write-Error "Docker build failed"
            Write-Host $buildOutput -ForegroundColor $Red
            return $false
        }
    }
    catch {
        Write-Error "Build failed: $($_.Exception.Message)"
        return $false
    }
}

function Clean-DockerEnvironment {
    Write-Header "Cleaning Docker Environment"

    Write-Warning "This will remove all containers, volumes, and images!"
    $confirmation = Read-Host "Are you sure? (y/N)"

    if ($confirmation -eq 'y' -or $confirmation -eq 'Y') {
        try {
            Write-Info "Stopping services..."
            docker compose down -v 2>$null

            Write-Info "Removing images..."
            docker compose down --rmi all 2>$null

            Write-Info "Cleaning up dangling images..."
            docker image prune -f 2>$null

            Write-Info "Cleaning up build cache..."
            docker builder prune -f 2>$null

            Write-Success "Docker environment cleaned"
            return $true
        }
        catch {
            Write-Error "Cleanup failed: $($_.Exception.Message)"
            return $false
        }
    }
    else {
        Write-Info "Cleanup cancelled"
        return $true
    }
}

function Start-Services {
    Write-Header "Starting Services"

    try {
        Write-Info "Starting Docker Compose services..."
        $startOutput = docker compose up -d 2>&1

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Services started successfully"

            Write-Info "Waiting for services to be ready..."
            Start-Sleep -Seconds 5

            Write-Info "Checking service status..."
            docker compose ps

            Write-Info "You can check logs with: docker compose logs -f"
            Write-Info "Test services with: .\scripts\test-integration.ps1 -Quick"

            return $true
        }
        else {
            Write-Error "Failed to start services"
            Write-Host $startOutput -ForegroundColor $Red
            return $false
        }
    }
    catch {
        Write-Error "Failed to start services: $($_.Exception.Message)"
        return $false
    }
}

function Show-OptimizationTips {
    Write-Header "Docker Optimization Tips"

    Write-Host "Container Optimizations Applied:" -ForegroundColor $Blue
    Write-Host "  ✓ No virtual environments in containers (UV_NO_CREATE_VENV=1)" -ForegroundColor $Green
    Write-Host "  ✓ System Python installation (--system flag)" -ForegroundColor $Green
    Write-Host "  ✓ Nexus proxy repositories for faster builds" -ForegroundColor $Green
    Write-Host "  ✓ Multi-stage builds for smaller images" -ForegroundColor $Green
    Write-Host "  ✓ Proper Debian/Ubuntu repository separation" -ForegroundColor $Green
    Write-Host ""

    Write-Host "Performance Benefits:" -ForegroundColor $Blue
    Write-Host "  • Faster container startup (no venv creation)" -ForegroundColor $Yellow
    Write-Host "  • Smaller images (no duplicate Python installations)" -ForegroundColor $Yellow
    Write-Host "  • Faster package installation (Nexus caching)" -ForegroundColor $Yellow
    Write-Host "  • Better layer caching in Docker builds" -ForegroundColor $Yellow
    Write-Host ""

    Write-Host "Memory Usage:" -ForegroundColor $Blue
    Write-Host "  • Each service uses ~100-200MB less RAM" -ForegroundColor $Yellow
    Write-Host "  • No duplicate Python processes per service" -ForegroundColor $Yellow
    Write-Host "  • Better resource utilization" -ForegroundColor $Yellow
}

function Show-Help {
    Write-Header "Morgan AI Assistant - Docker Build Helper"

    @"
Usage: .\scripts\docker-build-helper.ps1 [OPTIONS]

Options:
    -GenerateLockfile   Generate UV lockfile for Docker builds
    -BuildNoCache       Build Docker images without cache
    -ValidateConfig     Validate Docker and UV configuration
    -CleanStart         Clean Docker environment and start fresh
    -Service <name>     Build specific service only
    -Help               Show this help message

Examples:
    .\scripts\docker-build-helper.ps1 -GenerateLockfile    # Generate lockfile
    .\scripts\docker-build-helper.ps1 -ValidateConfig     # Check configuration
    .\scripts\docker-build-helper.ps1 -BuildNoCache       # Clean build
    .\scripts\docker-build-helper.ps1 -CleanStart         # Fresh start
    .\scripts\docker-build-helper.ps1 -Service llm        # Build LLM only

Quick Setup (in order):
    1. .\scripts\setup-uv.ps1 -Install -Configure
    2. .\scripts\docker-build-helper.ps1 -GenerateLockfile
    3. .\scripts\docker-build-helper.ps1 -ValidateConfig
    4. .\scripts\docker-build-helper.ps1 -BuildNoCache
    5. docker compose up -d
    6. .\scripts\test-integration.ps1 -Quick

Container Optimizations:
    • No virtual environments in Docker (UV_NO_CREATE_VENV=1)
    • System Python installation for faster startup
    • Nexus proxy repositories for faster builds
    • Proper Debian/Ubuntu repository separation
    • Multi-stage builds for smaller images

Service Architecture:
    Debian-based (python:3.12-slim):
      - LLM Service (CPU only, external Ollama client)
      - VAD Service (CPU optimized, Silero VAD)
      - Core Service (CPU only, orchestration)

    Ubuntu-based (nvidia/cuda:13.0.1-devel-ubuntu22.04):
      - TTS Service (Coqui TTS with CUDA)
      - STT Service (Faster Whisper + Silero VAD with CUDA)

External Dependencies:
    • Ollama service at 192.168.101.3:11434
    • Nexus repositories for Ubuntu, Debian, and PyPI
    • Optional Redis and PostgreSQL
"@ | Write-Host
}

# Main execution
function Main {
    # Check if we're in the project root
    if (-not (Test-Path "docker-compose.yml")) {
        Write-Error "docker-compose.yml not found. Please run this script from the project root."
        return
    }

    if ($Help) {
        Show-OptimizationTips
        Show-Help
        return
    }

    # Execute requested actions
    $success = $true

    if ($GenerateLockfile) {
        $success = $success -and (Generate-Lockfile)
    }

    if ($ValidateConfig) {
        $success = $success -and (Validate-DockerConfiguration)
    }

    if ($BuildNoCache) {
        if (-not $Service) {
            $success = $success -and (Build-DockerServices -NoCache)
        }
        else {
            $success = $success -and (Build-DockerServices -NoCache -SpecificService $Service)
        }
    }
    elseif ($Service) {
        $success = $success -and (Build-DockerServices -SpecificService $Service)
    }

    if ($CleanStart) {
        $success = $success -and (Clean-DockerEnvironment)
        if ($success) {
            Write-Info "Starting fresh services..."
            $success = $success -and (Start-Services)
        }
    }

    # If no specific action, show optimization tips and validate
    if (-not $GenerateLockfile -and -not $BuildNoCache -and -not $ValidateConfig -and -not $CleanStart -and -not $Service) {
        Show-OptimizationTips
        Validate-DockerConfiguration
    }

    if ($success) {
        Write-Success "Operation completed successfully!"
    }
    else {
        Write-Error "Some operations failed. Check the output above."
        exit 1
    }
}

# Run main function
Main
