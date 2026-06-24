[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Pythonw = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $Python)) {
    throw ".venv not found. Run scripts\setup_env.ps1 first."
}

# Preload the CUDA/cuDNN runtime DLLs bundled in the venv so the CUDA EP can start.
$CudnnBin = Join-Path $ProjectRoot ".venv\Lib\site-packages\nvidia\cudnn\bin"
$CublasBin = Join-Path $ProjectRoot ".venv\Lib\site-packages\nvidia\cublas\bin"
$env:PATH = "$CudnnBin;$CublasBin;$env:PATH"

Start-Process -FilePath $Pythonw -ArgumentList "-m", "panodeon.app" -WorkingDirectory $ProjectRoot
