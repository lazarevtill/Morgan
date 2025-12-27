@echo off
REM ============================================================================
REM Morgan AI Assistant - Setup and Deployment Script for Windows
REM
REM Usage:
REM   setup.bat [OPTIONS] [COMPONENTS...]
REM
REM Examples:
REM   setup.bat --core --db                    Start core services with databases
REM   setup.bat --all                          Start all services
REM   setup.bat --check-models                 Check Ollama models
REM   setup.bat --pull-models                  Download missing models
REM   setup.bat --core --env=.\custom.env      Use custom env file
REM   setup.bat --stop                         Stop all services
REM   setup.bat --status                       Show service status
REM
REM Docker Components:
REM   --core        Morgan core server (API, orchestration)
REM   --db          Databases (Qdrant + Redis)
REM   --reranking   Reranking service (CrossEncoder)
REM   --monitoring  Prometheus + Grafana
REM   --background  Background job processor
REM   --all         All components
REM
REM Ollama (Baremetal):
REM   --check-models    Check if required models are installed
REM   --pull-models     Download missing Ollama models
REM   --ollama-status   Show Ollama service status
REM
REM Options:
REM   --env=FILE    Specify environment file (default: .env)
REM   --build       Force rebuild containers
REM   --pull        Pull latest Docker images
REM   --stop        Stop services
REM   --restart     Restart services
REM   --status      Show service status
REM   --logs        View service logs
REM   --distributed Use distributed compose file
REM   -d            Run in background (default)
REM   -f            Run in foreground
REM   -h, --help    Show this help message
REM
REM Copyright 2025 Morgan AI Assistant Contributors
REM SPDX-License-Identifier: Apache-2.0
REM ============================================================================

setlocal EnableDelayedExpansion

REM ============================================================================
REM Configuration
REM ============================================================================

set "SCRIPT_DIR=%~dp0"
set "DOCKER_DIR=%SCRIPT_DIR%docker"
set "DEFAULT_ENV_FILE=%SCRIPT_DIR%.env"
set "COMPOSE_FILE=%DOCKER_DIR%\docker-compose.yml"
set "DISTRIBUTED_COMPOSE_FILE=%DOCKER_DIR%\docker-compose.distributed.yml"

REM Ollama endpoint (baremetal)
if not defined OLLAMA_HOST set "OLLAMA_HOST=http://localhost:11434"

REM Required models
set "REQUIRED_LLM_MODELS=qwen2.5:32b-instruct-q4_K_M qwen2.5:7b-instruct-q5_K_M"
set "REQUIRED_EMBEDDING_MODELS=qwen3-embedding:4b"

REM ============================================================================
REM Default values
REM ============================================================================

set "ENV_FILE=%DEFAULT_ENV_FILE%"
set "DETACH=true"
set "BUILD=false"
set "PULL=false"
set "USE_DISTRIBUTED=false"

REM Component flags
set "COMP_CORE=false"
set "COMP_DB=false"
set "COMP_RERANKING=false"
set "COMP_MONITORING=false"
set "COMP_BACKGROUND=false"
set "COMP_ALL=false"

REM Action flags
set "ACTION_START=true"
set "ACTION_STOP=false"
set "ACTION_RESTART=false"
set "ACTION_STATUS=false"
set "ACTION_LOGS=false"
set "ACTION_CHECK_MODELS=false"
set "ACTION_PULL_MODELS=false"
set "ACTION_OLLAMA_STATUS=false"
set "LOGS_SERVICE="

REM ============================================================================
REM Parse arguments
REM ============================================================================

:parse_args
if "%~1"=="" goto :end_parse

if /i "%~1"=="--core" (
    set "COMP_CORE=true"
    shift
    goto :parse_args
)
if /i "%~1"=="--db" (
    set "COMP_DB=true"
    shift
    goto :parse_args
)
if /i "%~1"=="--reranking" (
    set "COMP_RERANKING=true"
    shift
    goto :parse_args
)
if /i "%~1"=="--monitoring" (
    set "COMP_MONITORING=true"
    shift
    goto :parse_args
)
if /i "%~1"=="--background" (
    set "COMP_BACKGROUND=true"
    shift
    goto :parse_args
)
if /i "%~1"=="--all" (
    set "COMP_ALL=true"
    shift
    goto :parse_args
)
if /i "%~1"=="--build" (
    set "BUILD=true"
    shift
    goto :parse_args
)
if /i "%~1"=="--pull" (
    set "PULL=true"
    shift
    goto :parse_args
)
if /i "%~1"=="--distributed" (
    set "USE_DISTRIBUTED=true"
    shift
    goto :parse_args
)
if /i "%~1"=="-d" (
    set "DETACH=true"
    shift
    goto :parse_args
)
if /i "%~1"=="-f" (
    set "DETACH=false"
    shift
    goto :parse_args
)
if /i "%~1"=="--stop" (
    set "ACTION_START=false"
    set "ACTION_STOP=true"
    shift
    goto :parse_args
)
if /i "%~1"=="--restart" (
    set "ACTION_START=false"
    set "ACTION_RESTART=true"
    shift
    goto :parse_args
)
if /i "%~1"=="--status" (
    set "ACTION_START=false"
    set "ACTION_STATUS=true"
    shift
    goto :parse_args
)
if /i "%~1"=="--check-models" (
    set "ACTION_START=false"
    set "ACTION_CHECK_MODELS=true"
    shift
    goto :parse_args
)
if /i "%~1"=="--pull-models" (
    set "ACTION_START=false"
    set "ACTION_PULL_MODELS=true"
    shift
    goto :parse_args
)
if /i "%~1"=="--ollama-status" (
    set "ACTION_START=false"
    set "ACTION_OLLAMA_STATUS=true"
    shift
    goto :parse_args
)
if /i "%~1"=="--logs" (
    set "ACTION_START=false"
    set "ACTION_LOGS=true"
    shift
    if not "%~1"=="" (
        echo %~1 | findstr /b /c:"--" >nul || (
            set "LOGS_SERVICE=%~1"
            shift
        )
    )
    goto :parse_args
)
if /i "%~1"=="-h" goto :show_help
if /i "%~1"=="--help" goto :show_help

REM Parse --env=VALUE
echo %~1 | findstr /b /c:"--env=" >nul && (
    for /f "tokens=2 delims==" %%a in ("%~1") do set "ENV_FILE=%%a"
    shift
    goto :parse_args
)

echo [ERROR] Unknown option: %~1
echo Use --help for usage information
exit /b 1

:end_parse

REM ============================================================================
REM Show banner
REM ============================================================================

call :print_banner

REM ============================================================================
REM Create .env.example if needed
REM ============================================================================

call :create_env_example

REM ============================================================================
REM Load environment file
REM ============================================================================

call :load_env_file

REM ============================================================================
REM Execute action
REM ============================================================================

if "%ACTION_CHECK_MODELS%"=="true" (
    call :do_check_models
    goto :eof
)

if "%ACTION_PULL_MODELS%"=="true" (
    call :do_pull_models
    goto :eof
)

if "%ACTION_OLLAMA_STATUS%"=="true" (
    call :do_ollama_status
    goto :eof
)

if "%ACTION_STOP%"=="true" (
    call :check_docker
    if errorlevel 1 exit /b 1
    call :do_stop
    goto :eof
)

if "%ACTION_RESTART%"=="true" (
    call :check_docker
    if errorlevel 1 exit /b 1
    call :do_restart
    goto :eof
)

if "%ACTION_STATUS%"=="true" (
    call :do_status
    goto :eof
)

if "%ACTION_LOGS%"=="true" (
    call :check_docker
    if errorlevel 1 exit /b 1
    call :do_logs
    goto :eof
)

if "%ACTION_START%"=="true" (
    call :check_docker
    if errorlevel 1 exit /b 1

    REM Check if at least one component selected
    if "%COMP_CORE%"=="false" if "%COMP_DB%"=="false" if "%COMP_RERANKING%"=="false" if "%COMP_MONITORING%"=="false" if "%COMP_BACKGROUND%"=="false" if "%COMP_ALL%"=="false" (
        echo [WARN] No components specified. Use --help for options.
        echo [INFO] Starting default components: --core --db
        set "COMP_CORE=true"
        set "COMP_DB=true"
    )
    call :do_start
)

goto :eof

REM ============================================================================
REM Functions
REM ============================================================================

:print_banner
echo.
echo ========================================================================
echo.
echo    M   M  OOO  RRRR   GGG   AAA  N   N
echo    MM MM O   O R   R G     A   A NN  N
echo    M M M O   O RRRR  G  GG AAAAA N N N
echo    M   M O   O R  R  G   G A   A N  NN
echo    M   M  OOO  R   R  GGG  A   A N   N
echo.
echo              Personal AI Assistant - Self-Hosted
echo.
echo ========================================================================
echo.
goto :eof

:show_help
call :print_banner
echo Usage: setup.bat [OPTIONS] [COMPONENTS...]
echo.
echo Docker Components:
echo   --core        Morgan core server (API, orchestration)
echo   --db          Databases (Qdrant vector DB + Redis cache)
echo   --reranking   Reranking service (CrossEncoder)
echo   --monitoring  Monitoring stack (Prometheus + Grafana)
echo   --background  Background job processor
echo   --all         All Docker components
echo.
echo Ollama (Baremetal):
echo   --check-models    Check if required models are installed
echo   --pull-models     Download missing Ollama models
echo   --ollama-status   Show Ollama service status
echo.
echo Options:
echo   --env=FILE       Specify environment file (default: .env)
echo   --build          Force rebuild containers
echo   --pull           Pull latest Docker images before starting
echo   --distributed    Use distributed multi-host compose file
echo   -d               Run in background (default)
echo   -f               Run in foreground
echo.
echo Actions:
echo   --stop           Stop Docker services
echo   --restart        Restart Docker services
echo   --status         Show all service status
echo   --logs [SERVICE] View service logs (optional service name)
echo.
echo Examples:
echo   setup.bat --check-models                 Verify Ollama models
echo   setup.bat --pull-models                  Download missing models
echo   setup.bat --core --db                    Start core + databases
echo   setup.bat --all                          Start all Docker services
echo   setup.bat --core --env=.\prod.env        Use production env file
echo   setup.bat --stop                         Stop all services
echo   setup.bat --status                       Show complete status
echo.
echo Environment Variables:
echo   OLLAMA_HOST      Ollama API endpoint (default: http://localhost:11434)
echo.
exit /b 0

:check_docker
where docker >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop first.
    echo [INFO] Visit: https://docs.docker.com/desktop/install/windows-install/
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker daemon is not running.
    echo [INFO] Please start Docker Desktop and try again.
    exit /b 1
)

docker compose version >nul 2>&1
if errorlevel 1 (
    docker-compose version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Docker Compose is not available.
        echo [INFO] Please ensure Docker Desktop is properly installed.
        exit /b 1
    )
)

echo [INFO] Docker is available and running.
goto :eof

:check_ollama
echo [INFO] Checking Ollama at: %OLLAMA_HOST%

curl -sf "%OLLAMA_HOST%/api/version" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama is not running or not accessible at %OLLAMA_HOST%
    echo [INFO] Start Ollama with: ollama serve
    echo [INFO] Or check OLLAMA_HOST environment variable
    exit /b 1
)

echo [SUCCESS] Ollama is running
exit /b 0

:do_check_models
echo [INFO] Checking required Ollama models...
echo.

call :check_ollama
if errorlevel 1 exit /b 1

echo.
echo LLM Models:
set "missing_count=0"
set "installed_count=0"

for %%m in (%REQUIRED_LLM_MODELS%) do (
    call :check_single_model "%%m"
)

echo.
echo Embedding Models:
for %%m in (%REQUIRED_EMBEDDING_MODELS%) do (
    call :check_single_model "%%m"
)

echo.
echo ================================================================
echo Installed: !installed_count!  Missing: !missing_count!
echo ================================================================

if !missing_count! gtr 0 (
    echo.
    echo [WARN] Some required models are missing.
    echo [INFO] Run: setup.bat --pull-models  to download them
    exit /b 1
) else (
    echo [SUCCESS] All required models are installed!
)
goto :eof

:check_single_model
set "model=%~1"
REM Check if model is installed using ollama list
ollama list 2>nul | findstr /c:"%model%" >nul 2>&1
if errorlevel 1 (
    echo   [X] %model% ^(missing^)
    set /a "missing_count+=1"
) else (
    echo   [OK] %model%
    set /a "installed_count+=1"
)
goto :eof

:do_pull_models
echo [INFO] Downloading required Ollama models...
echo.

call :check_ollama
if errorlevel 1 exit /b 1

set "downloaded=0"
set "skipped=0"
set "failed=0"

for %%m in (%REQUIRED_LLM_MODELS% %REQUIRED_EMBEDDING_MODELS%) do (
    echo.
    ollama list 2>nul | findstr /c:"%%m" >nul 2>&1
    if errorlevel 1 (
        echo [INFO] Downloading: %%m
        echo   This may take a while depending on model size...
        ollama pull %%m
        if errorlevel 1 (
            echo [ERROR] Failed to download: %%m
            set /a "failed+=1"
        ) else (
            echo [SUCCESS] Downloaded: %%m
            set /a "downloaded+=1"
        )
    ) else (
        echo [INFO] Model already installed: %%m
        set /a "skipped+=1"
    )
)

echo.
echo ================================================================
echo Downloaded: !downloaded!  Skipped: !skipped!  Failed: !failed!
echo ================================================================

if !failed! gtr 0 (
    echo [ERROR] Some models failed to download
    exit /b 1
)

echo [SUCCESS] Model download complete!
goto :eof

:do_ollama_status
echo [INFO] Ollama Service Status
echo.

call :check_ollama
if errorlevel 1 goto :eof

echo.
echo Installed Models:
ollama list 2>nul
if errorlevel 1 (
    echo   ^(no models installed^)
)

echo.
echo Running Models:
ollama ps 2>nul
if errorlevel 1 (
    echo   ^(no models currently loaded^)
)
goto :eof

:load_env_file
if exist "%ENV_FILE%" (
    echo [INFO] Loading environment from: %ENV_FILE%
    for /f "usebackq tokens=1,* delims==" %%a in ("%ENV_FILE%") do (
        REM Skip comments and empty lines
        echo %%a | findstr /b /c:"#" >nul || (
            if not "%%a"=="" if not "%%b"=="" set "%%a=%%b"
        )
    )
    REM Update OLLAMA_HOST if set in env file
    if defined MORGAN_LLM_ENDPOINT set "OLLAMA_HOST=!MORGAN_LLM_ENDPOINT!"
) else (
    if not "%ENV_FILE%"=="%DEFAULT_ENV_FILE%" (
        echo [ERROR] Environment file not found: %ENV_FILE%
        exit /b 1
    ) else (
        echo [WARN] No .env file found. Using default configuration.
        echo [INFO] Copy .env.example to .env and configure as needed.
    )
)
goto :eof

:get_compose_file
if "%USE_DISTRIBUTED%"=="true" (
    set "ACTIVE_COMPOSE_FILE=%DISTRIBUTED_COMPOSE_FILE%"
) else (
    set "ACTIVE_COMPOSE_FILE=%COMPOSE_FILE%"
)
goto :eof

:build_services
set "SERVICES="
set "PROFILES="

if "%COMP_ALL%"=="true" (
    set "PROFILES=--profile monitoring --profile background"
    goto :eof_build_services
)

if "%COMP_CORE%"=="true" (
    set "SERVICES=!SERVICES! morgan-server"
    if "%USE_DISTRIBUTED%"=="true" set "SERVICES=!SERVICES! morgan-core"
)

if "%COMP_DB%"=="true" (
    set "SERVICES=!SERVICES! qdrant redis"
)

if "%COMP_RERANKING%"=="true" (
    if "%USE_DISTRIBUTED%"=="true" (
        set "PROFILES=!PROFILES! --profile gpu-reranking"
    )
)

if "%COMP_MONITORING%"=="true" (
    set "PROFILES=!PROFILES! --profile monitoring"
    set "SERVICES=!SERVICES! prometheus grafana"
)

if "%COMP_BACKGROUND%"=="true" (
    set "PROFILES=!PROFILES! --profile background"
    if "%USE_DISTRIBUTED%"=="true" set "SERVICES=!SERVICES! morgan-background"
)

:eof_build_services
goto :eof

:do_start
echo [INFO] Starting Morgan Docker services...

REM Check Ollama first
echo.
call :check_ollama
if not errorlevel 1 (
    echo.
    REM Quick model check - just warn if missing
    set "has_missing=false"
    for %%m in (%REQUIRED_LLM_MODELS% %REQUIRED_EMBEDDING_MODELS%) do (
        ollama list 2>nul | findstr /c:"%%m" >nul 2>&1
        if errorlevel 1 set "has_missing=true"
    )
    if "!has_missing!"=="true" (
        echo [WARN] Some required Ollama models are missing.
        echo [INFO] Run: setup.bat --check-models  to see details
        echo [INFO] Run: setup.bat --pull-models   to download them
        echo.
    )
) else (
    echo [WARN] Ollama is not running. LLM features will not work.
    echo.
)

call :get_compose_file
call :build_services

set "UP_ARGS=up"
if "%DETACH%"=="true" set "UP_ARGS=!UP_ARGS! -d"
if "%BUILD%"=="true" set "UP_ARGS=!UP_ARGS! --build"

set "COMPOSE_CMD=docker compose -f "%ACTIVE_COMPOSE_FILE%""
if exist "%ENV_FILE%" set "COMPOSE_CMD=!COMPOSE_CMD! --env-file "%ENV_FILE%""

if "%PULL%"=="true" (
    echo [INFO] Pulling latest Docker images...
    !COMPOSE_CMD! %PROFILES% pull
)

echo [INFO] Running: !COMPOSE_CMD! %PROFILES% %UP_ARGS% %SERVICES%
!COMPOSE_CMD! %PROFILES% %UP_ARGS% %SERVICES%

if "%DETACH%"=="true" (
    echo [SUCCESS] Docker services started in background.
    echo.
    call :do_status
)
goto :eof

:do_stop
echo [INFO] Stopping Morgan Docker services...

call :get_compose_file

docker compose -f "%ACTIVE_COMPOSE_FILE%" down

echo [SUCCESS] Docker services stopped.
goto :eof

:do_restart
echo [INFO] Restarting Morgan Docker services...
call :do_stop
timeout /t 2 /nobreak >nul
call :do_start
goto :eof

:do_status
echo.
echo ================================================================
echo                    Morgan Service Status
echo ================================================================
echo.

REM Ollama Status
echo Ollama ^(Baremetal^):
curl -sf "%OLLAMA_HOST%/api/version" >nul 2>&1
if errorlevel 1 (
    echo   Status:   NOT RUNNING
    echo   Endpoint: %OLLAMA_HOST%
) else (
    echo   Status:   RUNNING
    echo   Endpoint: %OLLAMA_HOST%
    for /f %%i in ('ollama list 2^>nul ^| find /c /v ""') do set "model_count=%%i"
    echo   Models:   !model_count! installed
)

echo.
echo Docker Services:

call :get_compose_file

docker compose -f "%ACTIVE_COMPOSE_FILE%" ps 2>nul || echo   ^(no services running^)

echo.
echo Health Checks:

REM Check Morgan Server
docker ps --format "{{.Names}}" 2>nul | findstr /c:"morgan-server" /c:"morgan-core" >nul && (
    curl -sf http://localhost:8080/health >nul 2>&1 && (
        echo   Morgan Server: HEALTHY
    ) || (
        echo   Morgan Server: UNHEALTHY
    )
)

REM Check Qdrant
docker ps --format "{{.Names}}" 2>nul | findstr /c:"qdrant" >nul && (
    curl -sf http://localhost:6333/healthz >nul 2>&1 && (
        echo   Qdrant:        HEALTHY
    ) || (
        echo   Qdrant:        UNHEALTHY
    )
)

REM Check Redis
docker ps --format "{{.Names}}" 2>nul | findstr /c:"redis" >nul && (
    docker exec morgan-redis redis-cli ping >nul 2>&1 && (
        echo   Redis:         HEALTHY
    ) || (
        echo   Redis:         UNHEALTHY
    )
)

echo.
goto :eof

:do_logs
call :get_compose_file

if not "%LOGS_SERVICE%"=="" (
    echo [INFO] Showing logs for: %LOGS_SERVICE%
    docker compose -f "%ACTIVE_COMPOSE_FILE%" logs -f %LOGS_SERVICE%
) else (
    echo [INFO] Showing logs for all services (Ctrl+C to exit)
    docker compose -f "%ACTIVE_COMPOSE_FILE%" logs -f
)
goto :eof

:create_env_example
if not exist "%SCRIPT_DIR%.env.example" (
    (
        echo # Morgan AI Assistant - Environment Configuration
        echo # Copy this file to .env and customize as needed
        echo.
        echo # =============================================================================
        echo # Ollama Configuration ^(Baremetal - not Docker^)
        echo # =============================================================================
        echo # Ollama endpoint - adjust if running on different host
        echo MORGAN_LLM_ENDPOINT=http://localhost:11434
        echo MORGAN_LLM_MODEL=qwen2.5:32b-instruct-q4_K_M
        echo MORGAN_LLM_FAST_MODEL=qwen2.5:7b-instruct-q5_K_M
        echo.
        echo # For distributed setup ^(comma-separated endpoints^)
        echo # MORGAN_LLM_ENDPOINTS=http://192.168.1.20:11434/v1,http://192.168.1.21:11434/v1
        echo MORGAN_LLM_STRATEGY=round_robin
        echo.
        echo # =============================================================================
        echo # Embedding Configuration ^(via Ollama^)
        echo # =============================================================================
        echo MORGAN_EMBEDDING_PROVIDER=ollama
        echo MORGAN_EMBEDDING_MODEL=qwen3-embedding:4b
        echo MORGAN_EMBEDDING_DIMENSIONS=2048
        echo # MORGAN_EMBEDDING_ENDPOINT=http://192.168.1.22:11434/v1
        echo.
        echo # =============================================================================
        echo # Reranking Configuration
        echo # =============================================================================
        echo MORGAN_RERANKING_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
        echo # MORGAN_RERANKING_ENDPOINT=http://192.168.1.23:8080/rerank
        echo.
        echo # =============================================================================
        echo # Database Configuration ^(Docker^)
        echo # =============================================================================
        echo # Redis
        echo MORGAN_REDIS_PASSWORD=
        echo MORGAN_REDIS_PREFIX=morgan:
        echo.
        echo # Qdrant
        echo MORGAN_VECTOR_DB_API_KEY=
        echo.
        echo # =============================================================================
        echo # Model Downloads ^(Optional^)
        echo # =============================================================================
        echo # Hugging Face token for gated models
        echo HF_TOKEN=
        echo.
        echo # =============================================================================
        echo # Monitoring ^(Optional^)
        echo # =============================================================================
        echo GRAFANA_ADMIN_USER=admin
        echo GRAFANA_ADMIN_PASSWORD=CHANGE_ME_IN_PRODUCTION
        echo.
        echo # =============================================================================
        echo # Performance
        echo # =============================================================================
        echo MORGAN_LOG_LEVEL=INFO
        echo MORGAN_LOG_FORMAT=json
        echo MORGAN_CACHE_SIZE_MB=1000
        echo MORGAN_MAX_CONCURRENT=100
        echo MORGAN_REQUEST_TIMEOUT=60
    ) > "%SCRIPT_DIR%.env.example"
    echo [INFO] Created .env.example file
)
goto :eof

endlocal
