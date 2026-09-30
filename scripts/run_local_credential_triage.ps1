param(
    [Parameter(Mandatory = $true)][string]$VaultRoot,
    [Parameter(Mandatory = $true)][string]$LogPath,
    [string]$Python = 'python',
    [string]$Model = 'qwen2.5:7b',
    [ValidateRange(1, 10000)][int]$MaxPages = 100,
    [ValidateRange(1, 100)][int]$PageSize = 60,
    [ValidateRange(0, 100)][int]$ModelLimit = 12
)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $VaultRoot -PathType Container)) {
    throw 'Vault root is missing'
}
if (Test-Path -LiteralPath $LogPath) {
    throw 'Log path already exists'
}
if (-not (Test-Path -LiteralPath (Split-Path -Parent $LogPath) -PathType Container)) {
    throw 'Log parent is missing'
}
if ($ModelLimit -gt $PageSize) {
    throw 'ModelLimit must not exceed PageSize'
}

Set-Location -LiteralPath $repository
Start-Transcript -LiteralPath $LogPath -ErrorAction Stop | Out-Null
try {
    & $Python -m scripts.triage_credential_ambiguity --root $VaultRoot `
        --model $Model --limit $PageSize --model-limit $ModelLimit `
        --max-pages $MaxPages --apply
    $triageExit = $LASTEXITCODE
    Write-Host ("MUNINN_TRIAGE_EXIT_CODE={0}" -f $triageExit)
} finally {
    Stop-Transcript | Out-Null
}
exit $triageExit
