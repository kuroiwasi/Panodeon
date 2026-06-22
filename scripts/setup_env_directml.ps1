[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Push-Location $ProjectRoot
try {
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $Python)) {
        & py -3 -m venv (Join-Path $ProjectRoot ".venv")
        if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv" }
    }
    & $Python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
    & $Python -m pip install -e ".[dev]"
    if ($LASTEXITCODE -ne 0) { throw "Failed to install project" }
    # Only one ONNX Runtime package may be present at a time.
    & $Python -m pip uninstall -y onnxruntime onnxruntime-directml onnxruntime-gpu
    & $Python -m pip install onnxruntime-directml
    if ($LASTEXITCODE -ne 0) { throw "Failed to install onnxruntime-directml" }
    Write-Host "DirectML/NVIDIA/AMD ONNX Runtime installed in .venv"
}
finally {
    Pop-Location
}
