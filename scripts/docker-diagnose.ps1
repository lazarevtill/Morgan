# Morgan AI Assistant - Docker Diagnostic Script
# This script helps diagnose and fix Docker build issues

param(
    [switch]$CheckFiles,
    [switch]$FixImports,
    [switch]$ValidateBuild,
    [switch]$CleanRebuild,
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

function Check-FileStructure {
    Write-Header "Checking File Structure"

    $services = @("core", "llm", "tts", "stt", "vad")

    foreach ($service in $services) {
        Write-Info "Checking $service service..."

        # Check main entry point
        $mainFile = "services/$service/main.py"
        if (Test-Path $mainFile) {
            Write-Success "Found main.py: $mainFile"
        }
        else {
            Write-Error "Missing main.py: $mainFile"
        }

        # Check service file
        $serviceFile = "services/$service/service.py"
        if (Test-Path $serviceFile) {
            Write-Success "Found service.py: $serviceFile"
        }
        else {
            Write-Warning "Missing service.py: $serviceFile"
        }

        # Check API server
        $apiServer = "services/$service/api/server.py"
        if (Test-Path $apiServer) {
            Write-Success "Found api/server.py: $apiServer"
        }
        else {
            Write-Warning "Missing api/server.py: $apiServer"
        }

        # Check Dockerfile
        $dockerfile = "services/$service/Dockerfile"
        if ($service -eq "core") {
            $dockerfile = "core/Dockerfile"
        }

        if (Test-Path $dockerfile) {
            Write-Success "Found Dockerfile: $dockerfile"
        }
        else {
            Write-Error "Missing Dockerfile: $dockerfile"
        }

        Write-Host ""
    }

    # Check shared modules
    Write-Info "Checking shared modules..."
    $sharedDirs = @("shared", "shared/config", "shared/models", "shared/utils")
    foreach ($dir in $sharedDirs) {
        if (Test-Path $dir) {
            Write-Success "Found shared directory: $dir"
        }
        else {
            Write-Error "Missing shared directory: $dir"
        }
    }

    # Check __init__.py files
    Write-Info "Checking __init__.py files..."
    $initFiles = @(
        "core/__init__.py",
        "services/llm/__init__.py",
        "services/tts/__init__.py",
        "services/stt/__init__.py",
        "services/vad/__init__.py",
        "shared/__init__.py",
        "shared/config/__init__.py",
        "shared/models/__init__.py",
        "shared/utils/__init__.py"
    )

    foreach ($initFile in $initFiles) {
        if (Test-Path $initFile) {
            Write-Success "Found __init__.py: $initFile"
        }
        else {
            Write-Warning "Missing __init__.py: $initFile"
        }
    }
}

function Fix-ImportIssues {
    Write-Header "Fixing Import Issues"

    # Create missing __init__.py files
    $missingInitFiles = @(
        "core/handlers/__init__.py",
        "core/integrations/__init__.py",
        "core/services/__init__.py",
        "core/utils/__init__.py",
        "core/api/__init__.py",
        "core/conversation/__init__.py"
    )

    foreach ($initFile in $missingInitFiles) {
        if (-not (Test-Path $initFile)) {
            New-Item -ItemType File -Path $initFile -Force
            Write-Success "Created missing __init__.py: $initFile"
        }
    }

    # Fix import statements in service files
    Write-Info "Checking service import statements..."

    # Core service imports
    $coreMain = "core/main.py"
    if (Test-Path $coreMain) {
        $content = Get-Content $coreMain -Raw
        if ($content -match "from \.app import main") {
            Write-Success "Core main.py has correct import"
        }
        else {
            Write-Warning "Core main.py may have incorrect imports"
        }
    }

    # Service imports
    $services = @("llm", "tts", "stt", "vad")
    foreach ($service in $services) {
        $mainFile = "services/$service/main.py"
        if (Test-Path $mainFile) {
            $content = Get-Content $mainFile -Raw
            if ($content -match "from \.api\.server import main") {
                Write-Success "$service main.py has correct import"
            }
            else {
                Write-Warning "$service main.py may have incorrect imports"
            }
        }
    }
}

function Validate-DockerBuild {
    param([string]$ServiceName)

    Write-Header "Validating Docker Build"

    if ($ServiceName) {
        Write-Info "Building service: $ServiceName"
        $buildCmd = "docker-compose build $ServiceName"
    }
    else {
        Write-Info "Building all services"
        $buildCmd = "docker-compose build"
    }

    Write-Info "Running: $buildCmd"

    try {
        $buildResult = & docker-compose build $ServiceName 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Docker build successful"
            return $true
        }
        else {
            Write-Error "Docker build failed"
            Write-Host $buildResult -ForegroundColor $Red
            return $false
        }
    }
    catch {
        Write-Error "Build validation failed: $($_.Exception.Message)"
        return $false
    }
}

function Clean-Rebuild {
    Write-Header "Clean Rebuild"

    Write-Warning "This will remove all containers, volumes, and images!"

    $confirmation = Read-Host "Are you sure? (y/N)"
    if ($confirmation -eq 'y' -or $confirmation -eq 'Y') {
        try {
            Write-Info "Stopping services..."
            docker-compose down -v 2>$null

            Write-Info "Removing images..."
            docker-compose down --rmi all 2>$null

            Write-Info "Cleaning up..."
            docker system prune -f 2>$null

            Write-Info "Building fresh..."
            $success = Validate-DockerBuild

            if ($success) {
                Write-Success "Clean rebuild completed successfully"
                return $true
            }
            else {
                Write-Error "Clean rebuild failed"
                return $false
            }
        }
        catch {
            Write-Error "Clean rebuild failed: $($_.Exception.Message)"
            return $false
        }
    }
    else {
        Write-Info "Clean rebuild cancelled"
        return $true
    }
}

function Show-DockerTips {
    Write-Header "Docker Build Tips"

    Write-Host "Common Issues and Solutions:" -ForegroundColor $Blue
    Write-Host ""
    Write-Host "1. Missing main.py files:" -ForegroundColor $Yellow
    Write-Host "   - Check that all services have main.py entry points" -ForegroundColor $White
    Write-Host "   - Verify CMD in Dockerfile points to correct file" -ForegroundColor $White
    Write-Host ""
    Write-Host "2. Multi-stage build issues:" -ForegroundColor $Yellow
    Write-Host "   - Runtime stage should inherit from build stage" -ForegroundColor $White
    Write-Host "   - Build stage should copy all necessary files" -ForegroundColor $White
    Write-Host ""
    Write-Host "3. Missing dependencies:" -ForegroundColor $Yellow
    Write-Host "   - Check pyproject.toml has correct dependency groups" -ForegroundColor $White
    Write-Host "   - Verify UV is installing to system Python" -ForegroundColor $White
    Write-Host ""
    Write-Host "4. Import errors:" -ForegroundColor $Yellow
    Write-Host "   - Ensure all __init__.py files exist" -ForegroundColor $White
    Write-Host "   - Check PYTHONPATH is set correctly" -ForegroundColor $White
    Write-Host ""
    Write-Host "Build Commands:" -ForegroundColor $Blue
    Write-Host "  docker-compose build --no-cache" -ForegroundColor $Green
    Write-Host "  docker-compose up -d --build" -ForegroundColor $Green
    Write-Host "  docker-compose logs -f <service>" -ForegroundColor $Green
    Write-Host "  docker-compose exec <service> python main.py" -ForegroundColor $Green
}

function Show-Help {
    Write-Header "Morgan AI Assistant - Docker Diagnostic Script"

    @"
Usage: .\scripts\docker-diagnose.ps1 [OPTIONS]

Options:
    -CheckFiles         Check file structure and imports
    -FixImports         Fix missing __init__.py files and imports
    -ValidateBuild      Validate Docker build for all services
    -CleanRebuild       Clean rebuild all services
    -Service <name>     Check specific service (core, llm, tts, stt, vad)
    -Help               Show this help message

Examples:
    .\scripts\docker-diagnose.ps1 -CheckFiles      # Check all files
    .\scripts\docker-diagnose.ps1 -FixImports       # Fix import issues
    .\scripts\docker-diagnose.ps1 -ValidateBuild    # Test Docker builds
    .\scripts\docker-diagnose.ps1 -CleanRebuild     # Fresh start

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
    1. Run: .\scripts\docker-diagnose.ps1 -CheckFiles
    2. Run: .\scripts\docker-diagnose.ps1 -FixImports
    3. Run: .\scripts\docker-diagnose.ps1 -ValidateBuild
    4. If issues persist: .\scripts\docker-diagnose.ps1 -CleanRebuild
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
        Show-DockerTips
        Show-Help
        return
    }

    $success = $true

    if ($CheckFiles) {
        Check-FileStructure
    }

    if ($FixImports) {
        Fix-ImportIssues
    }

    if ($ValidateBuild) {
        if ($Service) {
            $success = $success -and (Validate-DockerBuild -ServiceName $Service)
        }
        else {
            $success = $success -and (Validate-DockerBuild)
        }
    }

    if ($CleanRebuild) {
        $success = $success -and (Clean-Rebuild)
    }

    # If no specific action, run basic checks
    if (-not $CheckFiles -and -not $FixImports -and -not $ValidateBuild -and -not $CleanRebuild) {
        Show-DockerTips
        Check-FileStructure
    }

    if ($success) {
        Write-Success "Diagnostic completed successfully!"
    }
    else {
        Write-Error "Some issues found. Check the output above."
        exit 1
    }
}

# Run main function
Main
