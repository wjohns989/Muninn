<#
.SYNOPSIS
    Muninn Memory Server — Windows Launcher
.DESCRIPTION
    Starts the Muninn server and optionally the system tray application.
    Detects Python automatically from PATH, conda, or common install locations.
.PARAMETER Mode
    Launch mode: "server" (default), "tray", or "both"
.PARAMETER NoBrowser
    Skip opening the dashboard in the browser
.EXAMPLE
    .\launch_muninn.ps1
    .\launch_muninn.ps1 -Mode tray
    .\launch_muninn.ps1 -Mode both
#>

param(
    [ValidateSet("server", "tray", "both")]
    [string]$Mode = "server",
    [switch]$NoBrowser,
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

# --- Python Discovery ---
function Find-Python {
    $requested = $PythonPath
    if (-not $requested) { $requested = [Environment]::GetEnvironmentVariable("MUNINN_PYTHON_PATH", "Process") }
    if (-not $requested) { $requested = [Environment]::GetEnvironmentVariable("MUNINN_PYTHON_PATH", "User") }
    if ($requested) {
        if (-not (Test-Path -LiteralPath $requested -PathType Leaf)) {
            throw "MUNINN_PYTHON_PATH does not name an existing executable."
        }
        $resolved = (Resolve-Path -LiteralPath $requested).Path
        if ([IO.Path]::GetFileName($resolved) -notmatch '^python(?:3(?:\.\d+)?)?\.exe$') {
            throw "MUNINN_PYTHON_PATH must name a Python executable."
        }
        $version = & $resolved --version 2>&1
        if ($LASTEXITCODE -ne 0 -or $version -notmatch "Python 3\.(1[0-9]|[2-9][0-9])") {
            throw "MUNINN_PYTHON_PATH must be Python 3.10 or newer."
        }
        return $resolved
    }

    # Check if python is on PATH
    $pythonPath = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonPath) {
        $version = & $pythonPath.Source --version 2>&1
        if ($version -match "Python 3\.1[0-9]") {
            return $pythonPath.Source
        }
    }

    # 2. Check conda environments
    $condaPaths = @(
        "$env:USERPROFILE\miniconda3\python.exe",
        "$env:USERPROFILE\anaconda3\python.exe",
        "$env:LOCALAPPDATA\miniconda3\python.exe",
        "C:\ProgramData\miniconda3\python.exe"
    )
    foreach ($p in $condaPaths) {
        if (Test-Path $p) {
            $version = & $p --version 2>&1
            if ($version -match "Python 3\.1[0-9]") {
                return $p
            }
        }
    }

    # 3. Check common install locations
    $commonPaths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "C:\Python313\python.exe",
        "C:\Python312\python.exe",
        "C:\Python311\python.exe"
    )
    foreach ($p in $commonPaths) {
        if (Test-Path $p) { return $p }
    }

    return $null
}

# --- Ollama Check ---
function Test-Ollama {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434" -TimeoutSec 2 -ErrorAction SilentlyContinue
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Start-Ollama {
    Write-Host "[Muninn] Ollama not detected. Attempting to start..." -ForegroundColor Yellow
    $ollamaPath = Get-Command ollama -ErrorAction SilentlyContinue
    if ($ollamaPath) {
        Start-Process -FilePath $ollamaPath.Source -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 3
        if (Test-Ollama) {
            Write-Host "[Muninn] Ollama started successfully." -ForegroundColor Green
            return $true
        }
    }
    Write-Host "[Muninn] WARNING: Ollama not available. Server will start in degraded mode." -ForegroundColor Red
    Write-Host "[Muninn] Install Ollama from https://ollama.com and run: ollama pull nomic-embed-text" -ForegroundColor Red
    return $false
}

# --- Main ---
$effectiveToken = [Environment]::GetEnvironmentVariable("MUNINN_AUTH_TOKEN", "Process")
if (-not $effectiveToken) { $effectiveToken = [Environment]::GetEnvironmentVariable("MUNINN_SERVER_AUTH_TOKEN", "Process") }
if (-not $effectiveToken) { $effectiveToken = [Environment]::GetEnvironmentVariable("MUNINN_AUTH_TOKEN", "User") }
if (-not $effectiveToken) { $effectiveToken = [Environment]::GetEnvironmentVariable("MUNINN_SERVER_AUTH_TOKEN", "User") }
if (-not $effectiveToken) { throw "MUNINN_AUTH_TOKEN or MUNINN_SERVER_AUTH_TOKEN is required for a secure launch." }
if ($effectiveToken.Length -lt 32) {
    throw "MUNINN_AUTH_TOKEN must contain at least 32 characters."
}
$env:MUNINN_AUTH_TOKEN = $effectiveToken
$noAuth = [Environment]::GetEnvironmentVariable("MUNINN_NO_AUTH", "Process")
if (-not $noAuth) { $noAuth = [Environment]::GetEnvironmentVariable("MUNINN_NO_AUTH", "User") }
if ($noAuth -eq "1") {
    throw "MUNINN_NO_AUTH=1 disables authentication; refusing to launch."
}
$env:MUNINN_NO_AUTH = "0"
$env:MUNINN_DEV_MODE = "false"

Write-Host ""
Write-Host "  ███╗   ███╗██╗   ██╗███╗   ██╗██╗███╗   ██╗███╗   ██╗" -ForegroundColor Magenta
Write-Host "  ████╗ ████║██║   ██║████╗  ██║██║████╗  ██║████╗  ██║" -ForegroundColor Magenta
Write-Host "  ██╔████╔██║██║   ██║██╔██╗ ██║██║██╔██╗ ██║██╔██╗ ██║" -ForegroundColor Magenta
Write-Host "  ██║╚██╔╝██║██║   ██║██║╚██╗██║██║██║╚██╗██║██║╚██╗██║" -ForegroundColor Magenta
Write-Host "  ██║ ╚═╝ ██║╚██████╔╝██║ ╚████║██║██║ ╚████║██║ ╚████║" -ForegroundColor Magenta
Write-Host "  ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝" -ForegroundColor Magenta
Write-Host "       The Persistent Memory MCP — v3.0.0" -ForegroundColor DarkGray
Write-Host ""

# Find Python
$python = Find-Python
if (-not $python) {
    Write-Host "[Muninn] ERROR: Python 3.10+ not found." -ForegroundColor Red
    Write-Host "[Muninn] Install from https://python.org or conda." -ForegroundColor Red
    exit 1
}
Write-Host "[Muninn] Python: $python" -ForegroundColor Cyan

# Check Ollama
if (-not (Test-Ollama)) {
    Start-Ollama | Out-Null
}

# Set working directory
Set-Location $ProjectRoot

# Determine port
$port = if ($env:MUNINN_PORT) { $env:MUNINN_PORT } else { "42069" }

switch ($Mode) {
    "server" {
        Write-Host "[Muninn] Starting server on http://localhost:$port ..." -ForegroundColor Green
        if (-not $NoBrowser) {
            Start-Sleep -Seconds 3
            Start-Process "http://localhost:$port"
        }
        & $python server.py
    }
    "tray" {
        Write-Host "[Muninn] Starting system tray application..." -ForegroundColor Green
        # Use pythonw for windowless execution
        $pythonw = $python -replace "python\.exe$", "pythonw.exe"
        if (Test-Path $pythonw) {
            Start-Process -FilePath $pythonw -ArgumentList "tray_app.py" -WorkingDirectory $ProjectRoot -WindowStyle Hidden
        } else {
            Start-Process -FilePath $python -ArgumentList "tray_app.py" -WorkingDirectory $ProjectRoot -WindowStyle Hidden
        }
        Write-Host "[Muninn] Tray app launched. Look for the Muninn icon in your system tray." -ForegroundColor Cyan
    }
    "both" {
        Write-Host "[Muninn] Starting tray application (manages server lifecycle)..." -ForegroundColor Green
        # Tray app auto-starts the server
        $pythonw = $python -replace "python\.exe$", "pythonw.exe"
        if (Test-Path $pythonw) {
            Start-Process -FilePath $pythonw -ArgumentList "tray_app.py" -WorkingDirectory $ProjectRoot -WindowStyle Hidden
        } else {
            Start-Process -FilePath $python -ArgumentList "tray_app.py" -WorkingDirectory $ProjectRoot -WindowStyle Hidden
        }
        Write-Host "[Muninn] Tray app launched — server will start automatically." -ForegroundColor Cyan
        Write-Host "[Muninn] Dashboard: http://localhost:$port" -ForegroundColor Cyan
        if (-not $NoBrowser) {
            Start-Sleep -Seconds 5
            Start-Process "http://localhost:$port"
        }
    }
}
