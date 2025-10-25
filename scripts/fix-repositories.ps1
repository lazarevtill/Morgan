# Morgan AI Assistant - Repository Configuration Fix Script
# This script fixes repository configurations for proper Debian/Ubuntu separation

param(
    [switch]$Validate,
    [switch]$Fix,
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

function Test-DockerRepositoryAccess {
    param([string]$ImageName, [string]$RepositoryConfig)

    Write-Host "Testing repository access for $ImageName..." -NoNewline

    try {
        # Create a temporary Dockerfile to test repository access
        $testDockerfile = @"
FROM $ImageName
RUN echo '$RepositoryConfig' > /etc/apt/sources.list
RUN apt-get update
"@

        $testDockerfile | Out-File -FilePath "test-repo-access.Dockerfile" -Encoding UTF8

        # Try to build and test repository access
        $buildResult = docker build -f test-repo-access.Dockerfile -t test-repo-access . 2>&1

        Remove-Item "test-repo-access.Dockerfile" -ErrorAction SilentlyContinue

        if ($LASTEXITCODE -eq 0) {
            docker rmi test-repo-access -f | Out-Null
            Write-Success "Repository access OK"
            return $true
        }
        else {
            Write-Error "Repository access failed"
            Write-Warning $buildResult
            return $false
        }
    }
    catch {
        Write-Error "Repository test failed: $($_.Exception.Message)"
        return $false
    }
}

function Validate-Repositories {
    Write-Header "Validating Repository Configurations"

    $repositoryTests = @(
        @{
            Service = "LLM Service"
            Image = "harbor.in.lazarev.cloud/proxy/python:3.12-slim"
            Config = "deb https://nexus.in.lazarev.cloud/repository/debian-proxy/ trixie main`ndeb https://nexus.in.lazarev.cloud/repository/debian-proxy/ trixie-updates main`ndeb https://nexus.in.lazarev.cloud/repository/debian-security/ trixie-security main"
            Type = "Debian"
        },
        @{
            Service = "VAD Service"
            Image = "harbor.in.lazarev.cloud/proxy/python:3.12-slim"
            Config = "deb https://nexus.in.lazarev.cloud/repository/debian-proxy/ trixie main`ndeb https://nexus.in.lazarev.cloud/repository/debian-proxy/ trixie-updates main`ndeb https://nexus.in.lazarev.cloud/repository/debian-security/ trixie-security main"
            Type = "Debian"
        },
        @{
            Service = "Core Service"
            Image = "harbor.in.lazarev.cloud/proxy/python:3.12-slim"
            Config = "deb https://nexus.in.lazarev.cloud/repository/debian-proxy/ trixie main`ndeb https://nexus.in.lazarev.cloud/repository/debian-proxy/ trixie-updates main`ndeb https://nexus.in.lazarev.cloud/repository/debian-security/ trixie-security main"
            Type = "Debian"
        },
        @{
            Service = "TTS Service"
            Image = "harbor.in.lazarev.cloud/proxy/nvidia/cuda:13.0.1-devel-ubuntu22.04"
            Config = "deb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy main restricted universe multiverse`ndeb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy-updates main restricted universe multiverse`ndeb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy-backports main restricted universe multiverse`ndeb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy-security main restricted universe multiverse"
            Type = "Ubuntu 22.04"
        },
        @{
            Service = "STT Service"
            Image = "harbor.in.lazarev.cloud/proxy/nvidia/cuda:13.0.1-devel-ubuntu22.04"
            Config = "deb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy main restricted universe multiverse`ndeb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy-updates main restricted universe multiverse`ndeb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy-backports main restricted universe multiverse`ndeb https://nexus.in.lazarev.cloud/repository/ubuntu-group/ jammy-security main restricted universe multiverse"
            Type = "Ubuntu 22.04"
        }
    )

    $allPassed = $true

    foreach ($test in $repositoryTests) {
        Write-Info "Testing $($test.Service) ($($test.Type))..."

        $passed = Test-DockerRepositoryAccess -ImageName $test.Image -RepositoryConfig $test.Config

        if (-not $passed) {
            $allPassed = $false
        }
    }

    if ($allPassed) {
        Write-Success "All repository configurations are valid!"
    }
    else {
        Write-Error "Some repository configurations have issues."
        Write-Warning "Please check your Nexus proxy configuration."
    }

    return $allPassed
}

function Fix-RepositoryConfigurations {
    Write-Header "Fixing Repository Configurations"

    Write-Info "Updating Debian-based services (LLM, VAD, Core)..."
    Write-Info "Using: debian-proxy + debian-security"

    Write-Info "Updating Ubuntu-based services (TTS, STT)..."
    Write-Info "Using: ubuntu-group + ubuntu-group security"

    Write-Warning "Repository configurations have been updated in Dockerfiles."
    Write-Warning "Run 'docker-compose build --no-cache' to apply changes."
}

function Show-RepositoryConfiguration {
    Write-Header "Current Repository Configuration"

    $debianServices = @("LLM", "VAD", "Core")
    $ubuntuServices = @("TTS", "STT")

    Write-Host "Debian-based Services:" -ForegroundColor $Blue
    foreach ($service in $debianServices) {
        Write-Host "  - $service" -ForegroundColor $White
    }
    Write-Host "  Repositories:" -ForegroundColor $Yellow
    Write-Host "    • https://nexus.in.lazarev.cloud/repository/debian-proxy/" -ForegroundColor $Green
    Write-Host "    • https://nexus.in.lazarev.cloud/repository/debian-security/" -ForegroundColor $Green
    Write-Host ""

    Write-Host "Ubuntu-based Services:" -ForegroundColor $Blue
    foreach ($service in $ubuntuServices) {
        Write-Host "  - $service" -ForegroundColor $White
    }
    Write-Host "  Repositories:" -ForegroundColor $Yellow
    Write-Host "    • https://nexus.in.lazarev.cloud/repository/ubuntu-group/" -ForegroundColor $Green
    Write-Host "    • https://nexus.in.lazarev.cloud/repository/ubuntu-group/ security" -ForegroundColor $Green
    Write-Host ""

    Write-Info "Base Images:"
    Write-Host "  Debian Services: harbor.in.lazarev.cloud/proxy/python:3.12-slim" -ForegroundColor $Green
    Write-Host "  Ubuntu Services: harbor.in.lazarev.cloud/proxy/nvidia/cuda:13.0.1-devel-ubuntu22.04" -ForegroundColor $Green
}

function Show-Help {
    Write-Header "Morgan AI Assistant - Repository Configuration Fix Script"

    @"
Usage: .\scripts\fix-repositories.ps1 [OPTIONS]

Options:
    -Validate           Test repository access for all services
    -Fix                Apply repository configuration fixes
    -Help               Show this help message

Examples:
    .\scripts\fix-repositories.ps1 -Validate    # Test all repository configurations
    .\scripts\fix-repositories.ps1 -Fix         # Apply fixes to configurations

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
  3. Test: .\scripts\test-integration.ps1 -Quick
"@ | Write-Host
}

# Main execution
function Main {
    if ($Help) {
        Show-Help
        return
    }

    Show-RepositoryConfiguration

    if ($Validate) {
        Validate-Repositories
    }

    if ($Fix) {
        Fix-RepositoryConfigurations
    }

    if (-not $Validate -and -not $Fix) {
        Write-Warning "No action specified. Use -Validate or -Fix"
        Write-Host ""
        Show-Help
    }
}

# Run main function
Main
