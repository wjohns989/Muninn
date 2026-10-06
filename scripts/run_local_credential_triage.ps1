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
    [ValidateRange(1, 10000)][int]$MaxPages = 200,
    [ValidateRange(1, 100)][int]$PageSize = 60,
    [ValidateRange(0, 100)][int]$ModelLimit = 12,
    [ValidateRange(0, 86400)][int]$WaitForRemoteSeconds = 0
)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $VaultRoot -PathType Container)) {
    throw 'Vault root is missing'
}
if (-not (Test-Path -LiteralPath $ArchiveRoot -PathType Container)) {
    throw 'Archive root is missing'
}
if (Test-Path -LiteralPath $LogPath) {
    throw 'Log path already exists'
}
if (Test-Path -LiteralPath $BackupDestination) {
    throw 'Backup destination already exists'
}
if (Test-Path -LiteralPath $PreBackupDestination) {
    throw 'Pre-triage backup destination already exists'
}
if ($PreBackupDestination -eq $BackupDestination) {
    throw 'Pre- and post-triage backup destinations must differ'
}
if (-not (Test-Path -LiteralPath (Split-Path -Parent $LogPath) -PathType Container)) {
    throw 'Log parent is missing'
}
if (-not (Test-Path -LiteralPath (Split-Path -Parent $BackupDestination) -PathType Container)) {
    throw 'Backup parent is missing'
}
if (-not (Test-Path -LiteralPath (Split-Path -Parent $PreBackupDestination) -PathType Container)) {
    throw 'Pre-triage backup parent is missing'
}
if ($ModelLimit -gt $PageSize) {
    throw 'ModelLimit must not exceed PageSize'
}
if ($WaitForRemoteSeconds -gt 0 -and ($Provider -ne 'openrouter' -or $ModelLimit -eq 0)) {
    throw 'Remote readiness waiting requires OpenRouter model review'
}

Set-Location -LiteralPath $repository
if (-not $PolicyRoot) { $PolicyRoot = Split-Path -Parent $VaultRoot }
$modelArguments = @()
if ($Model) { $modelArguments = @('--model', $Model) }
$progressParent = if ($Provider -eq 'openrouter') { Join-Path $PolicyRoot 'remote_policy' } else { Split-Path -Parent $LogPath }
$progressLog = Join-Path (Join-Path $progressParent 'triage-progress') ((Split-Path -Leaf $LogPath) + '.progress.jsonl')
Start-Transcript -LiteralPath $LogPath -ErrorAction Stop | Out-Null
try {
    & $Python -m scripts.triage_credential_ambiguity --root $VaultRoot --archive-root $ArchiveRoot `
        --provider $Provider --policy-root $PolicyRoot @modelArguments --limit $PageSize --model-limit $ModelLimit `
        --max-pages $MaxPages --wait-for-readiness $WaitForRemoteSeconds --progress-log $progressLog --backup-before $PreBackupDestination `
        --backup-after $BackupDestination --apply
    $triageExit = $LASTEXITCODE
    Write-Host ("MUNINN_TRIAGE_EXIT_CODE={0}" -f $triageExit)
} finally {
    Stop-Transcript | Out-Null
}
exit $triageExit
