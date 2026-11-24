param([string]$Python = "C:/Users/OMEN/.virtualenvs/.venv/Scripts/python.exe")
$ErrorActionPreference = "Stop"
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskOldPath = $env:PYTHONPATH
$taskOldDirectory = Get-Location
try {
    Set-Location -LiteralPath $taskRoot
    $env:PYTHONPATH = "$taskRoot/.runtime/p0p1_torch;$taskRoot/.runtime/p0p1_shared"
    if ($taskOldPath) { $env:PYTHONPATH += ";$taskOldPath" }
    & $Python run_cf_formal_bridge.py validate
    if ($LASTEXITCODE -ne 0) { throw "P6-B bridge config validation failed" }
    & $Python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "P6-B bridge regressions failed" }
} finally {
    $env:PYTHONPATH = $taskOldPath
    Set-Location -LiteralPath $taskOldDirectory
}
