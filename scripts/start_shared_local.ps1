<#
.SYNOPSIS
Starts or verifies a single authenticated, loopback-only Muninn service.
.DESCRIPTION
Uses the user-scoped MUNINN_AUTH_TOKEN when the caller has not inherited it.
The script never prints the token or stores it in a launch argument or file.
#>
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonPath = "python",
    [string]$DataDir = "",
    [int]$Port = 42069,
    [int]$StartupTimeoutSeconds = 90
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot "server.py"))) {
    throw "Muninn server.py not found in the selected repository."
}
$pythonCommand = Get-Command $PythonPath -ErrorAction Stop
$PythonPath = $pythonCommand.Source
$token = [Environment]::GetEnvironmentVariable("MUNINN_AUTH_TOKEN", "User")
if (-not $token) {
    $token = [Environment]::GetEnvironmentVariable("MUNINN_AUTH_TOKEN", "Process")
}
if (-not $token) {
    throw "MUNINN_AUTH_TOKEN is missing. Set it in the Windows user environment first."
}

$baseUrl = "http://127.0.0.1:$Port"
$protectedUrl = "$baseUrl/profiles/model"
function Get-StatusCode([string]$Url, [hashtable]$Headers) {
    try {
        $params = @{ Uri = $Url; TimeoutSec = 10; UseBasicParsing = $true }
        if ($Headers) { $params.Headers = $Headers }
        return [int](Invoke-WebRequest @params).StatusCode
    } catch {
        if ($_.Exception.Response) { return [int]$_.Exception.Response.StatusCode }
        return 0
    }
}

function Assert-SecureServer {
    $withoutToken = Get-StatusCode $protectedUrl @{}
    $withToken = Get-StatusCode $protectedUrl @{ Authorization = "Bearer $token" }
    if ($withoutToken -ne 401 -or $withToken -ne 200) {
        throw "Muninn is listening but authentication is not verified (anonymous=$withoutToken, authenticated=$withToken)."
    }
    try {
        $history = Invoke-RestMethod -Uri "$baseUrl/history/status" -Headers @{ Authorization = "Bearer $token" } -TimeoutSec 15
    } catch {
        throw "Muninn is authenticated but history archive readiness could not be checked."
    }
    if ($history.data.history_security -ne "strict" -or $history.data.vault.mode -ne "strict" -or -not $history.data.vault.ready) {
        throw "Muninn is authenticated, but the strict encrypted history archive is not ready."
    }
}

$listeners = @(Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
if ($listeners.Count -gt 0) {
    Assert-SecureServer
    Write-Output "Muninn is already running securely at $baseUrl"
    exit 0
}
$otherListeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
if ($otherListeners.Count -gt 0) {
    throw "Port $Port has a non-loopback listener; refusing to start a duplicate service."
}

$env:MUNINN_AUTH_TOKEN = $token
$credentialApiToken = [Environment]::GetEnvironmentVariable("MUNINN_CREDENTIAL_API_TOKEN", "User")
if ($credentialApiToken) {
    $env:MUNINN_CREDENTIAL_API_TOKEN = $credentialApiToken
}
$env:MUNINN_NO_AUTH = "0"
$env:MUNINN_HOST = "127.0.0.1"
$env:MUNINN_PORT = [string]$Port
$env:MUNINN_HISTORY_SECURITY = "strict"
$env:MUNINN_CONSOLIDATION_DRY_RUN = "1"
$env:MUNINN_DEFER_LLM_ON_ADD = "true"
$env:MUNINN_OLLAMA_KEEP_ALIVE = "0"
$env:MUNINN_INSIGHTS_AUTO = "0"
$env:MUNINN_XLAM_ENABLED = "false"
$env:MUNINN_INSTRUCTOR_ENABLED = "false"
$env:MUNINN_RERANKER_ENABLED = "false"
foreach ($name in @("MUNINN_OLLAMA_MODEL", "MUNINN_AUTO_LOCAL_MODEL_HINTS")) {
    $userValue = [Environment]::GetEnvironmentVariable($name, "User")
    if ($userValue) { Set-Item -Path "Env:$name" -Value $userValue }
}
if ($DataDir) {
    $env:MUNINN_DATA_DIR = $DataDir
} elseif (-not $env:MUNINN_DATA_DIR) {
    $userDataDir = [Environment]::GetEnvironmentVariable("MUNINN_DATA_DIR", "User")
    $env:MUNINN_DATA_DIR = if ($userDataDir) { $userDataDir } else { ".muninn_runtime" }
}

$runtime = Join-Path $RepoRoot ".muninn_runtime"
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
Start-Process -FilePath $PythonPath -ArgumentList @("server.py") -WorkingDirectory $RepoRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $runtime "server-$stamp.stdout.log") `
    -RedirectStandardError (Join-Path $runtime "server-$stamp.stderr.log")

$deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
do {
    Start-Sleep -Seconds 1
    $listeners = @(Get-NetTCPConnection -LocalAddress "127.0.0.1" -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    if ($listeners.Count -gt 0) {
        Assert-SecureServer
        Write-Output "Muninn started securely at $baseUrl"
        exit 0
    }
} while ((Get-Date) -lt $deadline)
throw "Muninn did not start within $StartupTimeoutSeconds seconds. Inspect the private runtime logs."
