[CmdletBinding()]
param(
    [string]$Url = "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/488_DEIMv2-Wholebody49/resources.tar.gz",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if (-not $OutputDir) {
    $OutputDir = Join-Path $ProjectRoot "third_party\models"
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$Temp = Join-Path ([System.IO.Path]::GetTempPath()) ("models-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $Temp | Out-Null
try {
    $Archive = Join-Path $Temp "resources.tar.gz"
    Write-Host "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $Archive

    # tar.exe handles .tar.gz; -C extracts directly into the models folder.
    & tar.exe -xzf $Archive -C $OutputDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to extract $Archive"
    }
}
finally {
    if (Test-Path $Temp) {
        Remove-Item -Recurse -Force $Temp
    }
}
Write-Host "Downloaded to $OutputDir"
