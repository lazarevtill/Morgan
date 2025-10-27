# Morgan AI Services - Rebuild Script (PowerShell)
# This script rebuilds all services with fixes applied

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Morgan AI Services - Rebuild Script" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Stop all services
Write-Host "1. Stopping all services..." -ForegroundColor Yellow
docker-compose down

# Rebuild all services
Write-Host "2. Rebuilding all services..." -ForegroundColor Yellow
docker-compose build --no-cache

# Create necessary directories for model caching
Write-Host "3. Creating model cache directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data\models\huggingface" | Out-Null
New-Item -ItemType Directory -Force -Path "data\models\transformers" | Out-Null
New-Item -ItemType Directory -Force -Path "data\models\torch_hub\hub" | Out-Null
New-Item -ItemType Directory -Force -Path "data\models\sentence_transformers" | Out-Null
New-Item -ItemType Directory -Force -Path "data\models\tts" | Out-Null
New-Item -ItemType Directory -Force -Path "data\models\stt" | Out-Null
New-Item -ItemType Directory -Force -Path "logs\core" | Out-Null
New-Item -ItemType Directory -Force -Path "logs\llm" | Out-Null
New-Item -ItemType Directory -Force -Path "logs\tts" | Out-Null
New-Item -ItemType Directory -Force -Path "logs\stt" | Out-Null

# Start services
Write-Host "4. Starting services..." -ForegroundColor Yellow
docker-compose up -d

# Wait for services to start
Write-Host "5. Waiting for services to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check service health
Write-Host "6. Checking service health..." -ForegroundColor Yellow
Write-Host ""

Write-Host "Core Service:" -ForegroundColor Green
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -ErrorAction SilentlyContinue
    $response | ConvertTo-Json
} catch {
    Write-Host "Core service not ready" -ForegroundColor Red
}

Write-Host ""
Write-Host "LLM Service:" -ForegroundColor Green
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8001/health" -ErrorAction SilentlyContinue
    $response | ConvertTo-Json
} catch {
    Write-Host "LLM service not ready" -ForegroundColor Red
}

Write-Host ""
Write-Host "TTS Service:" -ForegroundColor Green
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8002/health" -ErrorAction SilentlyContinue
    $response | ConvertTo-Json
} catch {
    Write-Host "TTS service not ready" -ForegroundColor Red
}

Write-Host ""
Write-Host "STT Service:" -ForegroundColor Green
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8003/health" -ErrorAction SilentlyContinue
    $response | ConvertTo-Json
} catch {
    Write-Host "STT service not ready" -ForegroundColor Red
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Rebuild complete!" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "View logs with: docker-compose logs -f [service_name]" -ForegroundColor Yellow
Write-Host "Services:" -ForegroundColor Yellow
Write-Host "  - core (http://localhost:8000)" -ForegroundColor White
Write-Host "  - llm-service (http://localhost:8001)" -ForegroundColor White
Write-Host "  - tts-service (http://localhost:8002)" -ForegroundColor White
Write-Host "  - stt-service (http://localhost:8003)" -ForegroundColor White
Write-Host ""

