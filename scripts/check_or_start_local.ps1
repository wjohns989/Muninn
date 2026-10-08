<#
.SYNOPSIS
Check the shared local Muninn service, starting it only when absent.
.DESCRIPTION
Uses the secure single-instance launcher, then reports authenticated readiness
and nonsecret counts of capture and analysis problems. Never starts Ollama.
#>
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
try {
    $RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
    if (-not $PythonPath) {
        $PythonPath = [Environment]::GetEnvironmentVariable("MUNINN_PYTHON_PATH", "User")
    }
    if (-not $PythonPath) {
        $PythonPath = (Get-Command python -ErrorAction Stop).Source
    }
    $PythonPath = (Get-Command $PythonPath -ErrorAction Stop).Source

    & (Join-Path $RepoRoot "scripts\start_shared_local.ps1") -RepoRoot $RepoRoot -PythonPath $PythonPath
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "Muninn secure start/check failed."
    }

    $json = & $PythonPath -B (Join-Path $RepoRoot "scripts\local_runtime_preflight.py") `
        --authenticated --capture-errors --analysis-errors
    if ($LASTEXITCODE -ne 0) {
        throw "Muninn readiness inspection failed."
    }
    $report = $json | ConvertFrom-Json
    if ($report.health_http -ne 200 -or $report.anonymous_protected_http -ne 401 -or
        $report.authenticated_protected_http -ne 200 -or $report.history_http -ne 200 -or
        $report.history_security -ne "strict" -or -not $report.archive_ready -or
        $report.anonymous_root_contains_token) {
        throw "Muninn responded, but authentication or encrypted-history readiness is unhealthy."
    }
    Write-Host "Muninn is reachable, authenticated, and using encrypted history at http://127.0.0.1:42069" -ForegroundColor Green

    $captureProblems = @($report.capture_errors | Where-Object { $_ })
    $analysisProblems = @($report.analysis_errors | Where-Object { $_ })
    if ($captureProblems.Count -or $analysisProblems.Count) {
        Write-Warning "Muninn has historical or pending job problems to review:"
        foreach ($item in $captureProblems) {
            Write-Host ("  Capture: {0} {1} ({2}) x{3}" -f `
                $item.provider, $item.state, $item.error_code, $item.count)
        }
        foreach ($item in $analysisProblems) {
            Write-Host ("  Analysis: {0} x{1}" -f $item.state, $item.count)
        }
        exit 2
    }
    Write-Host "No outstanding capture or analysis job errors." -ForegroundColor Green
    exit 0
} catch {
    Write-Error $_.Exception.Message
    exit 1
}
