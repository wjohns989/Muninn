param(
    [Parameter(Mandatory = $true)][string]$VaultRoot,
    [Parameter(Mandatory = $true)][string]$ArchiveRoot,
    [Parameter(Mandatory = $true)][string]$BackupDestination,
    [Parameter(Mandatory = $true)][string]$LogPath,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $VaultRoot -PathType Container)) {
    throw 'Vault root is missing'
}
if (-not (Test-Path -LiteralPath $ArchiveRoot -PathType Container)) {
    throw 'Archive root is missing'
}
if (Test-Path -LiteralPath $BackupDestination) {
    throw 'Backup destination already exists'
}
if (Test-Path -LiteralPath $LogPath) {
    throw 'Log path already exists'
}
if (-not (Test-Path -LiteralPath (Split-Path -Parent $BackupDestination) -PathType Container)) {
    throw 'Backup parent is missing'
}
if (-not (Test-Path -LiteralPath (Split-Path -Parent $LogPath) -PathType Container)) {
    throw 'Log parent is missing'
}

Set-Location -LiteralPath $repository
Start-Transcript -LiteralPath $LogPath -ErrorAction Stop | Out-Null
try {
    & $Python -m muninn.cli credentials scan --root $VaultRoot `
        --archive-root $ArchiveRoot --backup-before $BackupDestination
    $scanExit = $LASTEXITCODE
    Write-Host ("MUNINN_SCAN_EXIT_CODE={0}" -f $scanExit)
} finally {
    Stop-Transcript | Out-Null
}
exit $scanExit
