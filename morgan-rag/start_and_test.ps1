# Start Morgan RAG and verify it's working

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Morgan RAG Setup and Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Change to morgan-rag directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
Write-Host "`nWorking directory: $(Get-Location)" -ForegroundColor Green

# Check if .env exists
if (-not (Test-Path .env)) {
    Write-Host "`nCreating .env file..." -ForegroundColor Yellow
    python create_env_file.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create .env file" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`.env file exists" -ForegroundColor Green
}

# Stop any existing containers
Write-Host "`nStopping existing containers..." -ForegroundColor Yellow
docker compose down 2>&1 | Out-Null

# Build and start services
Write-Host "`nBuilding and starting services..." -ForegroundColor Yellow
docker compose up -d --build

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to start services" -ForegroundColor Red
    exit 1
}

# Wait for services to be ready
Write-Host "`nWaiting for services to start (30 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Check container status
Write-Host "`nContainer Status:" -ForegroundColor Cyan
docker compose ps

# Check logs
Write-Host "`nRecent Morgan logs:" -ForegroundColor Cyan
docker compose logs --tail=30 morgan

# Test LLM connection
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Testing LLM Connection" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Wait a bit more for service to be fully ready
Start-Sleep -Seconds 10

# Test by making a request to the health endpoint or API
Write-Host "`nTesting service endpoints..." -ForegroundColor Yellow

try {
    $healthResponse = Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 10 -ErrorAction Stop
    Write-Host "Health check: $($healthResponse.StatusCode)" -ForegroundColor Green
} catch {
    Write-Host "Health endpoint not available yet, checking logs..." -ForegroundColor Yellow
    docker compose logs --tail=20 morgan
}

# Show final status
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Final Status" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
docker compose ps

Write-Host "`nServices should be running!" -ForegroundColor Green
Write-Host "Web interface: http://localhost:8080" -ForegroundColor Green
Write-Host "API: http://localhost:8000" -ForegroundColor Green
Write-Host "`nTo view logs: docker compose logs -f morgan" -ForegroundColor Yellow


