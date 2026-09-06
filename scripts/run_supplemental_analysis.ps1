param(
    [string]$PythonPath = $env:DISORDER_REPLICATION_PYTHON,
    [int]$BootstrapRepetitions = 199
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    if (Test-Path -LiteralPath 'C:\Softwares\miniconda3\envs\lym313\python.exe') {
        $PythonPath = 'C:\Softwares\miniconda3\envs\lym313\python.exe'
    } else {
        $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($null -ne $pythonCommand) {
            $PythonPath = $pythonCommand.Source
        } else {
            throw 'Python interpreter not found. Pass -PythonPath or set DISORDER_REPLICATION_PYTHON.'
        }
    }
}

& $PythonPath (Join-Path $projectRoot 'scripts\run_supplemental_analysis.py') --database (Join-Path $projectRoot 'data\current\disorder_payment_data.db') --output-dir (Join-Path $projectRoot 'outputs\supplemental') --bootstrap-repetitions $BootstrapRepetitions --seed 123
if ($LASTEXITCODE -ne 0) {
    throw "Supplemental analysis failed with exit code $LASTEXITCODE"
}
Write-Output "Supplemental analysis completed. Outputs: $(Join-Path $projectRoot 'outputs\supplemental')"
