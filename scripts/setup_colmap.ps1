[CmdletBinding()]
param(
    [string]$ColmapVersion = "4.0.4",
    [switch]$NoCuda,
    [string]$Dest = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if (-not $Dest) {
    $Dest = Join-Path $ProjectRoot "third_party\colmap"
}
$Dest = [System.IO.Path]::GetFullPath($Dest)
$ColmapExe = Join-Path $Dest "bin\colmap.exe"

if ((Test-Path $ColmapExe) -and -not $Force) {
    Write-Host "COLMAP already present at $Dest, skipping (use -Force to re-download)."
    return
}

$Variant = if ($NoCuda) { "nocuda" } else { "cuda" }
$Asset = "colmap-x64-windows-$Variant.zip"
$Url = "https://github.com/colmap/colmap/releases/download/$ColmapVersion/$Asset"

$Temp = Join-Path ([System.IO.Path]::GetTempPath()) ("colmap-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $Temp | Out-Null
try {
    $Archive = Join-Path $Temp "colmap.zip"
    Write-Host "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $Archive

    $Extract = Join-Path $Temp "extracted"
    New-Item -ItemType Directory -Force -Path $Extract | Out-Null
    # tar.exe (bsdtar) ships with Windows 10/11 and extracts zip safely.
    & tar.exe -xf $Archive -C $Extract
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to extract $Archive"
    }

    # COLMAP release zips wrap everything in a top-level colmap-x64-windows-* folder;
    # locate the directory that actually contains bin/colmap.exe.
    $Source = $null
    if (Test-Path (Join-Path $Extract "bin\colmap.exe")) {
        $Source = $Extract
    }
    else {
        foreach ($dir in Get-ChildItem -Path $Extract -Directory) {
            if (Test-Path (Join-Path $dir.FullName "bin\colmap.exe")) {
                $Source = $dir.FullName
                break
            }
        }
    }
    if (-not $Source) {
        throw "Could not locate bin/colmap.exe in the extracted archive under $Extract"
    }

    if (Test-Path $Dest) {
        Remove-Item -Recurse -Force $Dest
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Dest) | Out-Null
    Move-Item -Force $Source $Dest
}
finally {
    if (Test-Path $Temp) {
        Remove-Item -Recurse -Force $Temp
    }
}

if (-not (Test-Path $ColmapExe)) {
    throw "Expected COLMAP binary not found at $ColmapExe"
}
Write-Host "COLMAP $ColmapVersion installed to $Dest"
