@echo off
echo ========================================
echo Morgan RAG Setup and Start
echo ========================================
echo.

cd /d "%~dp0"

echo Checking .env file...
if not exist .env (
    echo Creating .env file...
    python create_env_file.py
    if errorlevel 1 (
        echo Failed to create .env file
        pause
        exit /b 1
    )
) else (
    echo .env file exists
)

echo.
echo Stopping existing containers...
docker compose down

echo.
echo Building and starting services...
docker compose up -d --build

if errorlevel 1 (
    echo Failed to start services
    pause
    exit /b 1
)

echo.
echo Waiting for services to start (45 seconds)...
timeout /t 45 /nobreak >nul

echo.
echo Container Status:
docker compose ps

echo.
echo Recent Morgan logs:
docker compose logs --tail=30 morgan

echo.
echo ========================================
echo Services should be running!
echo Web interface: http://localhost:8080
echo API: http://localhost:8000
echo ========================================
echo.
echo To view logs: docker compose logs -f morgan
echo To test LLM: python test_setup.py
echo.

pause


