# Experiment 2-2: local Qwen attention visualization (no LLM API key required).
# Full experiment: .\run-chapter2-attention.ps1
# Short prompt only: .\run-chapter2-attention.ps1 -Quick
[CmdletBinding()]
param(
    [ValidateSet('auto', 'cpu', 'cuda')]
    [string]$Device = 'auto',
    [switch]$Quick
)

$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Project Python is missing. Install the experiment dependencies first.'
}

$taskWorkspace = Split-Path -Parent $taskRoot
$env:HF_HOME = Join-Path $taskWorkspace '.cache\huggingface'
$env:HF_HUB_DISABLE_XET = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:HF_HUB_DOWNLOAD_TIMEOUT = '60'
$env:HF_HUB_ETAG_TIMEOUT = '30'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:MPLBACKEND = 'Agg'
$env:MPLCONFIGDIR = Join-Path $taskRoot '.local\matplotlib'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$taskArguments = @((Join-Path $taskRoot 'run-chapter2-attention.py'), '--device', $Device)
if ($Quick) { $taskArguments += '--quick' }
Push-Location -LiteralPath $taskRoot
try {
    & $taskPython @taskArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Attention experiment did not pass all checks (exit code $LASTEXITCODE). See its saved evidence."
    }
}
finally {
    Pop-Location
}
