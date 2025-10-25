# Morgan AI Assistant - UV Setup Script for Windows PowerShell
# This script helps configure UV for Windows development with Nexus repositories

param(
    [switch]$Install,
    [switch]$Configure,
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

function Install-UV {
    Write-Header "Installing UV Package Manager"

    Write-Info "Downloading and installing UV..."

    try {
        # Download UV installer
        $uvInstaller = "$env:TEMP\uv-installer.ps1"
        Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile $uvInstaller

        # Run installer
        & $uvInstaller

        # Add UV to PATH for current session
        $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"

        # Verify installation
        uv --version
        if ($LASTEXITCODE -eq 0) {
            Write-Success "UV installed successfully"
        }
        else {
            Write-Error "UV installation verification failed"
        }

        # Clean up installer
        Remove-Item $uvInstaller -ErrorAction SilentlyContinue
    }
    catch {
        Write-Error "Failed to install UV: $($_.Exception.Message)"
        Write-Warning "Please install UV manually from: https://github.com/astral-sh/uv"
    }
}

function Configure-UV {
    Write-Header "Configuring UV for Nexus Repositories"

    # Set UV environment variables
    Write-Info "Setting UV environment variables..."

    $env:UV_INDEX_URL = "https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple"
    $env:UV_COMPILE_BYTECODE = "1"
    $env:UV_LINK_MODE = "copy"

    # Update user environment variables (persistent)
    [Environment]::SetEnvironmentVariable("UV_INDEX_URL", "https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple", "User")
    [Environment]::SetEnvironmentVariable("UV_COMPILE_BYTECODE", "1", "User")
    [Environment]::SetEnvironmentVariable("UV_LINK_MODE", "copy", "User")

    Write-Success "UV configured for Nexus repositories"
    Write-Info "Index URL: $env:UV_INDEX_URL"
    Write-Info "Please restart PowerShell for changes to take effect"
}

function Validate-UVConfiguration {
    Write-Header "Validating UV Configuration"

    # Check if UV is installed
    try {
        $uvVersion = uv --version 2>$null
        Write-Success "UV is installed: $uvVersion"
    }
    catch {
        Write-Error "UV is not installed"
        return $false
    }

    # Check environment variables
    Write-Info "Checking UV environment variables..."

    $indexUrl = $env:UV_INDEX_URL
    if ($indexUrl -and $indexUrl.Contains("nexus.in.lazarev.cloud")) {
        Write-Success "UV_INDEX_URL is configured: $indexUrl"
    }
    else {
        Write-Warning "UV_INDEX_URL not configured or not using Nexus"
        Write-Info "Current UV_INDEX_URL: $indexUrl"
    }

    $compileBytecode = $env:UV_COMPILE_BYTECODE
    if ($compileBytecode -eq "1") {
        Write-Success "UV_COMPILE_BYTECODE is enabled"
    }
    else {
        Write-Warning "UV_COMPILE_BYTECODE not set to 1"
        Write-Info "Current UV_COMPILE_BYTECODE: $compileBytecode"
    }

    $linkMode = $env:UV_LINK_MODE
    if ($linkMode -eq "copy") {
        Write-Success "UV_LINK_MODE is set to copy"
    }
    else {
        Write-Warning "UV_LINK_MODE not set to copy"
        Write-Info "Current UV_LINK_MODE: $linkMode"
    }

    # Note: For local development, we use virtual environments since system Python is protected
    # Docker containers use UV_NO_CREATE_VENV=1 for system installation
    Write-Info "UV configured for both local (venv) and Docker (system) development"

    # Test UV system installation
    Write-Info "Testing UV system installation..."
    try {
        Push-Location $PSScriptRoot\..
        uv pip install --dry-run fastapi --system 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "UV system installation works"
        }
        else {
            Write-Warning "UV system installation may have issues"
        }
    }
    catch {
        Write-Warning "UV system installation test failed: $($_.Exception.Message)"
    }
    finally {
        Pop-Location
    }

    return $true
}

function Fix-CommonIssues {
    Write-Header "Fixing Common UV Issues"

    # Check if we're in the project directory
    if (-not (Test-Path "pyproject.toml")) {
        Write-Error "pyproject.toml not found. Please run this script from the project root."
        return
    }

    # Fix common pyproject.toml issues
    Write-Info "Checking pyproject.toml for common issues..."

    $pyprojectContent = Get-Content "pyproject.toml" -Raw

    # Check for invalid UV configuration
    if ($pyprojectContent -match "\[tool\.uv\.index\]") {
        Write-Warning "Found invalid [tool.uv.index] section in pyproject.toml"
        Write-Info "Removing invalid UV configuration..."

        # Remove the invalid section
        $fixedContent = $pyprojectContent -replace "\[tool\.uv\.index\][\s\S]*?(?=\[|$)", ""

        # Make sure we don't have duplicate sections
        $lines = $fixedContent -split "`n"
        $outputLines = @()
        $skipNext = $false

        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($skipNext) {
                $skipNext = $false
                continue
            }

            if ($lines[$i] -match "^\[tool\.uv\]") {
                # Check if next non-empty line is also [tool.uv]
                for ($j = $i + 1; $j -lt $lines.Count; $j++) {
                    if ($lines[$j].Trim() -eq "") {
                        continue
                    }
                    elseif ($lines[$j] -match "^\[tool\.uv\]") {
                        $skipNext = $true
                        break
                    }
                    else {
                        break
                    }
                }
            }

            if (-not $skipNext) {
                $outputLines += $lines[$i]
            }
        }

        $fixedContent = $outputLines -join "`n"

        # Write back to file
        $fixedContent | Out-File -FilePath "pyproject.toml" -Encoding UTF8
        Write-Success "Fixed pyproject.toml"
    }

    # Validate the fix
    try {
        Push-Location $PSScriptRoot\..
        uv sync --dry-run 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "pyproject.toml is now valid"
        }
        else {
            Write-Warning "pyproject.toml may still have issues"
        }
    }
    catch {
        Write-Warning "Could not validate pyproject.toml: $($_.Exception.Message)"
    }
    finally {
        Pop-Location
    }

    Write-Info "Common issues fixed. Try building again."
}

function Show-UVHelp {
    Write-Header "UV Setup Help"

    @"
UV (Ultra Fast Python Package Manager) Setup:

This script helps configure UV for Windows development with Nexus repositories.

Usage: .\scripts\setup-uv.ps1 [OPTIONS]

Options:
    -Install            Install UV package manager
    -Configure          Configure UV for Nexus repositories
    -Validate           Validate UV configuration and setup
    -Fix                Fix common UV and pyproject.toml issues
    -Help               Show this help message

Examples:
    .\scripts\setup-uv.ps1 -Install         # Install UV
    .\scripts\setup-uv.ps1 -Configure       # Configure for Nexus
    .\scripts\setup-uv.ps1 -Validate        # Check configuration
    .\scripts\setup-uv.ps1 -Fix             # Fix common issues

Configuration:
    UV_INDEX_URL: https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple
    UV_COMPILE_BYTECODE: 1 (compile Python files)
    UV_LINK_MODE: copy (copy packages instead of symlinks)

Development Strategy:
    - Local development: Uses virtual environments (system Python protected)
    - Docker containers: Uses system Python (UV_NO_CREATE_VENV=1)
    - Both use Nexus PyPI proxy for faster installs

Common Issues Fixed:
    - Invalid [tool.uv.index] sections in pyproject.toml
    - Missing UV environment variables
    - pyproject.toml parsing errors
    - Repository configuration problems

Docker Optimizations:
    - UV_NO_CREATE_VENV=1 (no virtual environment creation in containers)
    - No virtual environments created in Docker containers
    - Faster container builds and startup

Manual Configuration:
    1. Install UV: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    2. Set environment variables:
       [Environment]::SetEnvironmentVariable("UV_INDEX_URL", "https://nexus.in.lazarev.cloud/repository/pypi-proxy/simple", "User")
       [Environment]::SetEnvironmentVariable("UV_COMPILE_BYTECODE", "1", "User")
       [Environment]::SetEnvironmentVariable("UV_LINK_MODE", "copy", "User")
    3. Restart PowerShell
    4. Test: uv --version

For more information, visit: https://github.com/astral-sh/uv
"@ | Write-Host
}

# Main execution
function Main {
    # Check if UV is already installed
    $uvInstalled = $false
    try {
        $uvVersion = uv --version 2>$null
        $uvInstalled = $true
        Write-Info "UV is already installed: $uvVersion"
    }
    catch {
        Write-Info "UV is not installed"
    }

    # Execute requested actions
    if ($Help) {
        Show-UVHelp
        return
    }

    if ($Install) {
        Install-UV
    }

    if ($Configure) {
        Configure-UV
    }

    if ($Validate) {
        $success = Validate-UVConfiguration
        if (-not $success) {
            Write-Warning "UV configuration has issues. Run -Fix to resolve."
        }
    }

    if ($Fix) {
        Fix-CommonIssues
    }

    # If no specific action requested, show status
    if (-not $Install -and -not $Configure -and -not $Validate -and -not $Fix) {
        Validate-UVConfiguration
    }
}

# Run main function
Main
