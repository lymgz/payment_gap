param(
    [string]$PythonPath = $env:DISORDER_REPLICATION_PYTHON,
    [string]$SourceDir = ''
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($SourceDir)) {
    $upmRoot = Split-Path -Parent (Split-Path -Parent $projectRoot)
    $SourceDir = Join-Path $upmRoot 'debt crisis\Figure and table\code'
}

if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = 'C:\Softwares\miniconda3\envs\lym313\python.exe'
}
if ($PythonPath -ne 'python' -and -not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python interpreter not found: $PythonPath. Pass -PythonPath or set DISORDER_REPLICATION_PYTHON."
}

Set-Location -LiteralPath $projectRoot
& $PythonPath -m src.run_pipeline `
    --input-dir (Join-Path $projectRoot 'data\legacy') `
    --output-dir (Join-Path $projectRoot 'outputs\legacy') `
    --source-dir $SourceDir `
    --run-name legacy `
    --seed 123

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Output ('Legacy replication completed. Outputs: ' + (Join-Path $projectRoot 'outputs\legacy'))
