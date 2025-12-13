@echo off
REM Validation script for Docker Compose configuration (Windows)

echo === Morgan Docker Compose Validation ===
echo.

REM Check if docker-compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo X docker-compose is not installed
    exit /b 1
)
echo + docker-compose is installed

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo X Docker daemon is not running
    exit /b 1
)
echo + Docker daemon is running

REM Validate docker-compose.yml syntax
echo.
echo Validating docker-compose.yml...
docker-compose -f docker-compose.yml config --quiet >nul 2>&1
if %errorlevel% neq 0 (
    echo X docker-compose.yml has syntax errors
    exit /b 1
)
echo + docker-compose.yml is valid

REM Check if required files exist
echo.
echo Checking required files...
if exist docker-compose.yml (
    echo + docker-compose.yml exists
) else (
    echo X docker-compose.yml not found
    exit /b 1
)

if exist prometheus.yml (
    echo + prometheus.yml exists
) else (
    echo X prometheus.yml not found
    exit /b 1
)

if exist Dockerfile.server (
    echo + Dockerfile.server exists
) else (
    echo ! Dockerfile.server not found (will be needed for build^)
)

REM Check if .env file exists
echo.
if exist .env (
    echo + .env file exists
    echo.
    echo Environment variables configured:
    findstr /B "MORGAN_" .env 2>nul
) else (
    echo ! .env file not found
    echo   Copy .env.example to .env and configure your settings
)

echo.
echo === Validation Complete ===
echo.
echo To start services:
echo   docker-compose up -d
echo.
echo To start with monitoring:
echo   docker-compose --profile monitoring up -d
echo.
echo To view logs:
echo   docker-compose logs -f
echo.
