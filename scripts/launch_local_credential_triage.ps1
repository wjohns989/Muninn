<#
.SYNOPSIS
Opens an interactive local unlock window and leaves a nonsecret log to monitor.
.DESCRIPTION
No passphrase is supplied through arguments, environment, chat or a log.
The visible console is intentional: the user enters the existing passphrase.
#>
param(
    [Parameter(Mandatory = $true)][string]$VaultRoot,
    [Parameter(Mandatory = $true)][string]$ArchiveRoot,
    [Parameter(Mandatory = $true)][string]$LogPath,
    [Parameter(Mandatory = $true)][string]$PreBackupDestination,
    [Parameter(Mandatory = $true)][string]$BackupDestination,
    [string]$Python = 'python',
    [ValidateSet('openrouter', 'ollama')][string]$Provider = 'openrouter',
    [string]$PolicyRoot = '',
    [string]$Model = '',
    [ValidateRange(1, 10000)][int]$MaxPages = 1,
    [ValidateRange(1, 100)][int]$PageSize = 6,
    [ValidateRange(0, 100)][int]$ModelLimit = 2
)

$ErrorActionPreference = 'Stop'
$worker = Join-Path $PSScriptRoot 'run_local_credential_triage.ps1'
foreach ($root in @($VaultRoot, $ArchiveRoot)) {
    if (-not (Test-Path -LiteralPath $root -PathType Container)) { throw 'Private root missing' }
}
foreach ($destination in @($LogPath, $PreBackupDestination, $BackupDestination)) {
    if (Test-Path -LiteralPath $destination) { throw 'Destination already exists' }
    if (-not (Test-Path -LiteralPath (Split-Path -Parent $destination) -PathType Container)) {
        throw 'Destination parent missing'
    }
}
if ($ModelLimit -gt $PageSize) { throw 'ModelLimit must not exceed PageSize' }
if (-not $PolicyRoot) { $PolicyRoot = Split-Path -Parent $VaultRoot }
function Quote-Argument([string]$value) {
    if ($value.Contains('"') -or $value.Contains("`n") -or $value.Contains("`r")) {
        throw 'Unsupported argument character'
    }
    return '"' + $value.TrimEnd('\') + '"'
}
$arguments = @('-NoProfile', '-NoExit', '-File', (Quote-Argument $worker),
    '-VaultRoot', (Quote-Argument $VaultRoot), '-ArchiveRoot', (Quote-Argument $ArchiveRoot),
    '-LogPath', (Quote-Argument $LogPath), '-PreBackupDestination', (Quote-Argument $PreBackupDestination),
    '-BackupDestination', (Quote-Argument $BackupDestination), '-Python', (Quote-Argument $Python),
    '-Provider', $Provider, '-PolicyRoot', (Quote-Argument $PolicyRoot), '-MaxPages', [string]$MaxPages,
    '-PageSize', [string]$PageSize, '-ModelLimit', [string]$ModelLimit)
if ($Model) { $arguments += @('-Model', (Quote-Argument $Model)) }
# This is an interactive unlock, not a background helper. The console remains
# open after completion so the user can see the safe final status.
$process = Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments `
    -WindowStyle Normal -PassThru
Write-Output ('Opened local unlock window; nonsecret progress log: {0}' -f $LogPath)
