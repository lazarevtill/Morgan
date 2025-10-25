# Morgan AI Assistant Development Setup Script for Windows
# This script sets up the development environment for Morgan AI Assistant

param(
    [switch]$Build,
    [switch]$Up,
    [switch]$Down,
    [switch]$Clean,
    [switch]$Test,
    [switch]$Logs,
    [string]$Service,
    [switch]$Help
)

# Configuration
$COMPOSE_FILE = "docker-compose.yml"
$PROJECT_ROOT = $PSScriptRoot | Split-Path -Parent
$COMPOSE_CMD = "docker compose"

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

function Test-Prerequisites {
    Write-Header "Checking Prerequisites"

    # Check if Docker is running
    try {
        $dockerVersion = docker version
        Write-Success "Docker is installed and running"
    }
    catch {
        Write-Error "Docker is not installed or not running. Please install Docker Desktop."
        exit 1
    }

    # Check if docker-compose is available
    try {
        $composeVersion = docker compose version
        Write-Success "Docker Compose is available"
    }
    catch {
        Write-Error "Docker Compose is not available. Please update Docker Desktop."
        exit 1
    }

    # Check if NVIDIA Container Toolkit is available (for CUDA services)
    try {
        $nvidiaInfo = docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi 2>$null
        Write-Success "NVIDIA Container Toolkit is available"
    }
    catch {
        Write-Warning "NVIDIA Container Toolkit is not available. CUDA services will run on CPU."
    }

    # Check repository configurations
    Write-Info "Checking repository configurations..."
    try {
        & "$PSScriptRoot\fix-repositories.ps1" -Validate
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Repository configurations are valid"
        }
        else {
            Write-Warning "Repository configurations need attention. Run: .\scripts\fix-repositories.ps1 -Fix"
        }
    }
    catch {
        Write-Warning "Could not validate repository configurations: $($_.Exception.Message)"
    }

    Write-Info "Prerequisites check completed"
}

function Initialize-Environment {
    Write-Header "Initializing Environment"

    # Create necessary directories
    $directories = @(
        "logs/llm",
        "logs/tts",
        "logs/stt",
        "logs/vad",
        "logs/core",
        "data/models/llm",
        "data/models/tts",
        "data/models/stt",
        "data/voices",
        "data/conversations"
    )

    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force
            Write-Success "Created directory: $dir"
        }
        else {
            Write-Info "Directory already exists: $dir"
        }
    }

    # Create virtual environment for local development
    Write-Info "Creating virtual environment for local development..."
    try {
        uv venv
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Virtual environment created successfully"

            # Install dependencies in virtual environment
            Write-Info "Installing dependencies in virtual environment..."
            & ".\.venv\Scripts\activate.ps1"
            uv pip install fastapi uvicorn[standard] pydantic aiohttp pyyaml python-dotenv structlog psutil redis
            uv pip install pytest pytest-asyncio httpx pytest-cov black isort flake8

            if ($LASTEXITCODE -eq 0) {
                Write-Success "Dependencies installed in virtual environment"
            }
            else {
                Write-Warning "Failed to install some dependencies"
            }
        }
        else {
            Write-Warning "Failed to create virtual environment"
        }
    }
    catch {
        Write-Warning "Virtual environment creation failed: $($_.Exception.Message)"
    }
}

    # Create .env file if it doesn't exist
    if (-not (Test-Path ".env")) {
        @"
# Morgan AI Assistant Environment Configuration
# Copy this file to .env and modify as needed

# Ollama Configuration (external service)
OLLAMA_HOST=192.168.101.3:11434

# Service Configuration
MORGAN_CONFIG_DIR=./config

# CUDA Configuration
CUDA_VISIBLE_DEVICES=0

# Logging
LOG_LEVEL=INFO

# External Services (uncomment if needed)
# REDIS_URL=redis://localhost:6379
# POSTGRES_URL=postgresql://morgan:morgan_password@localhost:5432/morgan
"@ | Out-File -FilePath ".env" -Encoding UTF8
        Write-Success "Created .env template file"
        Write-Warning "Please edit .env file with your specific configuration"
    }
    else {
        Write-Info ".env file already exists"
    }
}

function Build-Services {
    Write-Header "Building Services"

    try {
        & $COMPOSE_CMD build --parallel
        if ($LASTEXITCODE -eq 0) {
            Write-Success "All services built successfully"
        }
        else {
            Write-Error "Failed to build services"
            exit 1
        }
    }
    catch {
        Write-Error "Build failed: $_"
        exit 1
    }
}

function Start-Services {
    Write-Header "Starting Services"

    try {
        & $COMPOSE_CMD up -d
        if ($LASTEXITCODE -eq 0) {
            Write-Success "All services started successfully"
            Write-Info "Services are starting up. This may take a few minutes..."
            Write-Info "Check status with: docker compose ps"
            Write-Info "View logs with: .\scripts\dev-setup.ps1 -Logs"
        }
        else {
            Write-Error "Failed to start services"
            exit 1
        }
    }
    catch {
        Write-Error "Failed to start services: $_"
        exit 1
    }
}

function Stop-Services {
    Write-Header "Stopping Services"

    try {
        & $COMPOSE_CMD down
        if ($LASTEXITCODE -eq 0) {
            Write-Success "All services stopped successfully"
        }
        else {
            Write-Error "Failed to stop services"
        }
    }
    catch {
        Write-Error "Failed to stop services: $_"
    }
}

function Clean-Environment {
    Write-Header "Cleaning Environment"

    Write-Warning "This will remove all containers, volumes, and images!"

    $confirmation = Read-Host "Are you sure? (y/N)"
    if ($confirmation -eq 'y' -or $confirmation -eq 'Y') {
        try {
            & $COMPOSE_CMD down -v --rmi all
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Environment cleaned successfully"

                # Remove directories
                $directories = @("logs", "data")
                foreach ($dir in $directories) {
                    if (Test-Path $dir) {
                        Remove-Item -Recurse -Force $dir
                        Write-Info "Removed directory: $dir"
                    }
                }
            }
            else {
                Write-Error "Failed to clean environment"
            }
        }
        catch {
            Write-Error "Failed to clean environment: $_"
        }
    }
    else {
        Write-Info "Clean operation cancelled"
    }
}

function Show-Logs {
    param([string]$ServiceName)

    if ($ServiceName) {
        Write-Header "Showing logs for service: $ServiceName"
        & $COMPOSE_CMD logs -f $ServiceName
    }
    else {
        Write-Header "Showing logs for all services"
        & $COMPOSE_CMD logs -f
    }
}

function Show-Status {
    Write-Header "Service Status"

    try {
        & $COMPOSE_CMD ps
    }
    catch {
        Write-Error "Failed to get service status: $_"
    }
}

function Run-Tests {
    Write-Header "Running Tests"

    try {
        # Run tests in a container or locally
        Write-Info "Running tests (this feature is not yet implemented)"
        Write-Info "You can run tests manually with: docker compose exec <service> pytest"
    }
    catch {
        Write-Error "Failed to run tests: $_"
    }
}

function Show-Help {
    Write-Header "Morgan AI Assistant Development Setup Script"

    @"
Usage: .\scripts\dev-setup.ps1 [OPTIONS]

Options:
    -Build              Build all Docker images
    -Up                 Start all services
    -Down               Stop all services
    -Clean              Clean environment (removes all containers, volumes, and images)
    -Test               Run tests (not yet implemented)
    -Logs               Show logs for all services
    -Service <name>     Specify service name for logs (requires -Logs)
    -Help               Show this help message

Examples:
    .\scripts\dev-setup.ps1 -Build -Up    # Build and start all services
    .\scripts\dev-setup.ps1 -Down         # Stop all services
    .\scripts\dev-setup.ps1 -Logs -Service llm  # Show logs for LLM service
    .\scripts\dev-setup.ps1 -Clean        # Clean entire environment

Prerequisites:
    - Docker Desktop with Docker Compose
    - NVIDIA Container Toolkit (optional, for CUDA services)
    - PowerShell 5.1 or higher

Configuration:
    Edit the .env file in the project root to configure:
    - Ollama service URL
    - CUDA device settings
    - Logging levels
    - External service connections

Development Strategy:
    - Local development: Uses virtual environments (.venv) since system Python is protected
    - Docker containers: Uses system Python (UV_NO_CREATE_VENV=1) for optimal performance
    - Both approaches use Nexus PyPI proxy for faster package installation

Services:
    - llm-service    : OpenAI-compatible client for external Ollama
    - tts-service    : Text-to-Speech service with CUDA support
    - stt-service    : Speech-to-Text service with CUDA and Silero VAD
    - vad-service    : Voice Activity Detection service (CPU optimized)
    - core           : Main orchestration service
    - redis          : Caching and message queue (optional)
    - postgres       : Persistent storage (optional)

For more information, visit: https://github.com/yourusername/morgan
"@ | Write-Host
}

# Main execution
function Main {
    # Change to project root directory
    Set-Location $PROJECT_ROOT

    # Show help if no arguments provided or -Help flag
    if ($args.Length -eq 0 -or $Help) {
        Show-Help
        return
    }

    # Execute requested actions
    if ($Clean) {
        Clean-Environment
    }
    elseif ($Down) {
        Stop-Services
    }
    else {
        # Check prerequisites
        Test-Prerequisites

        # Initialize environment
        Initialize-Environment

        # Build if requested
        if ($Build) {
            Build-Services
        }

        # Start if requested
        if ($Up) {
            Start-Services
        }

        # Show logs if requested
        if ($Logs) {
            Show-Logs -ServiceName $Service
        }

        # Show status
        Show-Status

        # Run tests if requested
        if ($Test) {
            Run-Tests
        }
    }
}

# Run main function
Main
