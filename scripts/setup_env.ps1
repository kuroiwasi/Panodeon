[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Push-Location $ProjectRoot
try {
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $Python)) {
        & python -m venv (Join-Path $ProjectRoot ".venv")
        if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv" }
    }
    & $Python -m pip install -e ".[dev,inference]"
    if ($LASTEXITCODE -ne 0) { throw "Failed to install project" }
    Write-Host "Setup complete."
}
finally {
    Pop-Location
}
