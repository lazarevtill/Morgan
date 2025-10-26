# Morgan AI Assistant - Private Registry Build Script
# This script builds and pushes images to the private Harbor registry

param(
    [switch]$Build,
    [switch]$Push,
    [switch]$Pull,
    [switch]$Clean,
    [string]$Service,
    [string]$Tag = "latest",
    [switch]$All,
    [switch]$Help
)

# Configuration
$REGISTRY_BASE = "harbor.in.lazarev.cloud/morgan"
$REGISTRY_PROXY = "harbor.in.lazarev.cloud/proxy"

# Service configurations
$services = @{
    "core" = @{
        "dockerfile" = "core/Dockerfile"
        "context" = "."
        "image" = "core"
    }
    "llm" = @{
        "dockerfile" = "services/llm/Dockerfile"
        "context" = "."
        "image" = "llm-service"
    }
    "tts" = @{
        "dockerfile" = "services/tts/Dockerfile"
        "context" = "."
        "image" = "tts-service"
    }
    "stt" = @{
        "dockerfile" = "services/stt/Dockerfile"
        "context" = "."
        "image" = "stt-service"
    }
    "vad" = @{
        "dockerfile" = "services/vad/Dockerfile"
        "context" = "."
        "image" = "vad-service"
    }
}

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

function Test-DockerRegistryAccess {
    Write-Header "Testing Docker Registry Access"

    try {
        # Test connection to registry
        Write-Info "Testing connection to $REGISTRY_BASE..."
        docker pull $REGISTRY_PROXY/python:3.12-slim 2>$null

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Registry access successful"
            docker rmi $REGISTRY_PROXY/python:3.12-slim -f | Out-Null
            return $true
        }
        else {
            Write-Error "Cannot access private registry"
            Write-Warning "Please ensure you are logged in to harbor.in.lazarev.cloud"
            Write-Info "Run: docker login harbor.in.lazarev.cloud"
            return $false
        }
    }
    catch {
        Write-Error "Registry access test failed: $($_.Exception.Message)"
        return $false
    }
}

function Build-ServiceImage {
    param([string]$ServiceName)

    $serviceConfig = $services[$ServiceName]
    if (-not $serviceConfig) {
        Write-Error "Unknown service: $ServiceName"
        return $false
    }

    Write-Header "Building $ServiceName Service"

    $imageName = "$REGISTRY_BASE/$($serviceConfig.image):$Tag"

    Write-Info "Building image: $imageName"
    Write-Info "Dockerfile: $($serviceConfig.dockerfile)"
    Write-Info "Context: $($serviceConfig.context)"

    try {
        $buildCmd = "docker build -t $imageName -f $($serviceConfig.dockerfile) $($serviceConfig.context)"
        Write-Info "Running: $buildCmd"

        $buildResult = Invoke-Expression $buildCmd 2>&1

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Successfully built $imageName"
            return $true
        }
        else {
            Write-Error "Build failed for $ServiceName"
            Write-Host $buildResult -ForegroundColor $Red
            return $false
        }
    }
    catch {
        Write-Error "Build failed for $ServiceName`: $($_.Exception.Message)"
        return $false
    }
}

function Push-ServiceImage {
    param([string]$ServiceName)

    $serviceConfig = $services[$ServiceName]
    if (-not $serviceConfig) {
        Write-Error "Unknown service: $ServiceName"
        return $false
    }

    Write-Header "Pushing $ServiceName Service"

    $imageName = "$REGISTRY_BASE/$($serviceConfig.image):$Tag"

    try {
        Write-Info "Pushing image: $imageName"
        docker push $imageName

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Successfully pushed $imageName"
            return $true
        }
        else {
            Write-Error "Push failed for $imageName"
            return $false
        }
    }
    catch {
        Write-Error "Push failed for $ServiceName`: $($_.Exception.Message)"
        return $false
    }
}

function Pull-ServiceImage {
    param([string]$ServiceName)

    $serviceConfig = $services[$ServiceName]
    if (-not $serviceConfig) {
        Write-Error "Unknown service: $ServiceName"
        return $false
    }

    Write-Header "Pulling $ServiceName Service"

    $imageName = "$REGISTRY_BASE/$($serviceConfig.image):$Tag"

    try {
        Write-Info "Pulling image: $imageName"
        docker pull $imageName

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Successfully pulled $imageName"
            return $true
        }
        else {
            Write-Warning "Pull failed for $imageName (image may not exist in registry)"
            return $false
        }
    }
    catch {
        Write-Error "Pull failed for $ServiceName`: $($_.Exception.Message)"
        return $false
    }
}

function Build-AllServices {
    Write-Header "Building All Services"

    $success = $true
    $serviceList = if ($Service) { @($Service) } else { $services.Keys }

    foreach ($serviceName in $serviceList) {
        $success = $success -and (Build-ServiceImage -ServiceName $serviceName)
        if (-not $success) {
            Write-Warning "Stopping build due to failure in $serviceName"
            break
        }
    }

    return $success
}

function Push-AllServices {
    Write-Header "Pushing All Services"

    $success = $true
    $serviceList = if ($Service) { @($Service) } else { $services.Keys }

    foreach ($serviceName in $serviceList) {
        $success = $success -and (Push-ServiceImage -ServiceName $serviceName)
        if (-not $success) {
            Write-Warning "Stopping push due to failure in $serviceName"
            break
        }
    }

    return $success
}

function Pull-AllServices {
    Write-Header "Pulling All Services"

    $success = $true
    $serviceList = if ($Service) { @($Service) } else { $services.Keys }

    foreach ($serviceName in $serviceList) {
        $success = $success -and (Pull-ServiceImage -ServiceName $serviceName)
    }

    return $success
}

function Show-RegistryInfo {
    Write-Header "Private Registry Information"

    Write-Host "Registry Base: $REGISTRY_BASE" -ForegroundColor $Blue
    Write-Host "Registry Proxy: $REGISTRY_PROXY" -ForegroundColor $Blue
    Write-Host ""

    Write-Host "Service Images:" -ForegroundColor $Blue
    foreach ($service in $services.Keys) {
        $imageName = "$REGISTRY_BASE/$($services[$service].image):$Tag"
        Write-Host "  $service -> $imageName" -ForegroundColor $Green
    }
    Write-Host ""

    Write-Host "Base Images Used:" -ForegroundColor $Blue
    Write-Host "  Python: $REGISTRY_PROXY/python:3.12-slim" -ForegroundColor $Yellow
    Write-Host "  CUDA: $REGISTRY_PROXY/nvidia/cuda:13.0.1-devel-ubuntu22.04" -ForegroundColor $Yellow
    Write-Host "  UV: harbor.in.lazarev.cloud/gh-proxy/astral-sh/uv:latest" -ForegroundColor $Yellow
    Write-Host "  Redis: $REGISTRY_PROXY/redis:7-alpine" -ForegroundColor $Yellow
    Write-Host "  PostgreSQL: $REGISTRY_PROXY/postgres:17-alpine" -ForegroundColor $Yellow
    Write-Host ""

    Write-Host "Commands:" -ForegroundColor $Blue
    Write-Host "  docker login harbor.in.lazarev.cloud" -ForegroundColor $Green
    Write-Host "  .\scripts\registry-build.ps1 -Build -Push" -ForegroundColor $Green
    Write-Host "  docker-compose pull" -ForegroundColor $Green
    Write-Host "  docker-compose up -d" -ForegroundColor $Green
}

function Show-Help {
    Write-Header "Morgan AI Assistant - Private Registry Build Script"

    @"
Usage: .\scripts\registry-build.ps1 [OPTIONS]

Options:
    -Build              Build images locally
    -Push               Push images to private registry
    -Pull               Pull images from private registry
    -Clean              Clean up local images after push
    -Service <name>     Build/push specific service (core, llm, tts, stt, vad)
    -Tag <tag>          Image tag (default: latest)
    -All                Build all services
    -Help               Show this help message

Examples:
    .\scripts\registry-build.ps1 -Build -Push         # Build and push all services
    .\scripts\registry-build.ps1 -Service core -Push  # Build and push core only
    .\scripts\registry-build.ps1 -Pull                # Pull all images from registry
    .\scripts\registry-build.ps1 -Clean               # Clean up after successful push

Registry Workflow:
    1. Login: docker login harbor.in.lazarev.cloud
    2. Build: .\scripts\registry-build.ps1 -Build
    3. Push: .\scripts\registry-build.ps1 -Push
    4. Deploy: docker-compose up -d

Available Services:
    - core     : Main orchestration service
    - llm      : OpenAI-compatible LLM client
    - tts      : Text-to-speech with CUDA
    - stt      : Speech-to-text with CUDA + VAD
    - vad      : Voice activity detection (CPU)

Image Naming Convention:
    harbor.in.lazarev.cloud/morgan/{service}:latest
    - harbor.in.lazarev.cloud/morgan/core:latest
    - harbor.in.lazarev.cloud/morgan/llm-service:latest
    - harbor.in.lazarev.cloud/morgan/tts-service:latest
    - harbor.in.lazarev.cloud/morgan/stt-service:latest
    - harbor.in.lazarev.cloud/morgan/vad-service:latest

Base Images:
    - Python services: harbor.in.lazarev.cloud/proxy/python:3.12-slim
    - CUDA services: harbor.in.lazarev.cloud/proxy/nvidia/cuda:13.0.1-devel-ubuntu22.04
    - UV: harbor.in.lazarev.cloud/gh-proxy/astral-sh/uv:latest (public)
    - Redis: harbor.in.lazarev.cloud/proxy/redis:7-alpine
    - PostgreSQL: harbor.in.lazarev.cloud/proxy/postgres:17-alpine

For troubleshooting, check:
    - Registry connectivity: docker login harbor.in.lazarev.cloud
    - Network access: curl https://harbor.in.lazarev.cloud
    - Permissions: Ensure you have push permissions to the morgan project
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
        Show-RegistryInfo
        Show-Help
        return
    }

    # Test registry access first
    if (-not (Test-DockerRegistryAccess)) {
        Write-Error "Cannot access private registry. Please login first."
        Write-Info "Run: docker login harbor.in.lazarev.cloud"
        return
    }

    $success = $true

    if ($Build) {
        if ($All -or (-not $Service)) {
            $success = $success -and (Build-AllServices)
        }
        else {
            $success = $success -and (Build-ServiceImage -ServiceName $Service)
        }
    }

    if ($Push) {
        if ($All -or (-not $Service)) {
            $success = $success -and (Push-AllServices)
        }
        else {
            $success = $success -and (Push-ServiceImage -ServiceName $Service)
        }
    }

    if ($Pull) {
        $success = $success -and (Pull-AllServices)
    }

    if ($Clean) {
        Write-Header "Cleaning Up Local Images"
        try {
            foreach ($service in $services.Keys) {
                $imageName = "$REGISTRY_BASE/$($services[$service].image):$Tag"
                docker rmi $imageName 2>$null | Out-Null
                Write-Info "Removed local image: $imageName"
            }
            Write-Success "Cleanup completed"
        }
        catch {
            Write-Warning "Some images may not have been removed: $($_.Exception.Message)"
        }
    }

    # Show registry information if no specific action
    if (-not $Build -and -not $Push -and -not $Pull -and -not $Clean) {
        Show-RegistryInfo
    }

    if ($success) {
        Write-Success "Registry operation completed successfully!"
    }
    else {
        Write-Error "Some operations failed. Check the output above."
        exit 1
    }
}

# Run main function
Main
