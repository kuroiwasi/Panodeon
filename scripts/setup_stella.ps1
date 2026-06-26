[CmdletBinding()]
param(
    [string]$ThirdPartyRoot = "",
    [switch]$SkipDependencyInstall
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$StellaCommit = "8ac1be4d1fa20e4148b478d7b30788abcbb1d9fe"
$ExamplesCommit = "defc69eecc36e51cdda22885bb86954f08ad6887"
$VcpkgCommit = "8cdd1410ea4e2b65c7e5176e63237240460886a6"
$VocabSha256 = "310CDF3581B5DD5BECFABD75823606E315B1F049F70E51FA23BE05021DC4B107"
$Triplet = "x64-windows-static"

$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if (-not $ThirdPartyRoot) {
    $ThirdPartyRoot = Join-Path $ProjectRoot "third_party"
}
$ThirdPartyRoot = [System.IO.Path]::GetFullPath($ThirdPartyRoot)
$StellaRoot = Join-Path $ThirdPartyRoot "stella_vslam"
$ExamplesRoot = Join-Path $ThirdPartyRoot "stella_vslam_examples"
$VcpkgRoot = Join-Path $StellaRoot "3rd\vcpkg"
$VocabRoot = Join-Path $ThirdPartyRoot "FBoW_orb_vocab"
$RuntimeRoot = Join-Path $ThirdPartyRoot "runtime"

function Invoke-Native {
    if ($args.Count -lt 1) {
        throw "Native executable is required"
    }
    $Executable = [string]$args[0]
    $NativeArguments = @()
    for ($Index = 1; $Index -lt $args.Count; $Index++) {
        $NativeArguments += $args[$Index]
    }
    & $Executable @NativeArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable failed with exit code $LASTEXITCODE"
    }
}

function Ensure-Repository {
    param(
        [string]$Url,
        [string]$Path,
        [string]$Commit
    )
    if (-not (Test-Path (Join-Path $Path ".git"))) {
        Invoke-Native git clone --filter=blob:none $Url $Path
        Invoke-Native git -C $Path fetch --depth 1 origin $Commit
        Invoke-Native git -C $Path checkout --detach $Commit
        Invoke-Native git -C $Path submodule update --init --recursive
    }
    $Current = (& git -C $Path rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $Current -ne $Commit) {
        throw "Unexpected revision in $Path`: $Current; expected $Commit"
    }
}

function Ensure-Patch {
    param(
        [string]$Repository,
        [string]$Patch
    )
    & git -C $Repository apply --reverse --check $Patch 2>$null
    if ($LASTEXITCODE -eq 0) {
        return
    }
    Invoke-Native git -C $Repository apply --check $Patch
    Invoke-Native git -C $Repository apply $Patch
}

New-Item -ItemType Directory -Force -Path $ThirdPartyRoot, $RuntimeRoot | Out-Null
Ensure-Repository "https://github.com/stella-cv/stella_vslam.git" $StellaRoot $StellaCommit
Ensure-Repository "https://github.com/stella-cv/stella_vslam_examples.git" $ExamplesRoot $ExamplesCommit
Ensure-Patch $StellaRoot (Join-Path $ProjectRoot "patches\stella_vslam-windows.patch")
Ensure-Patch $ExamplesRoot (Join-Path $ProjectRoot "patches\stella_vslam_examples-windows.patch")

if (-not (Test-Path (Join-Path $VcpkgRoot ".git"))) {
    Invoke-Native git clone https://github.com/microsoft/vcpkg.git $VcpkgRoot
    Invoke-Native git -C $VcpkgRoot fetch --depth 1 origin $VcpkgCommit
    Invoke-Native git -C $VcpkgRoot checkout --detach $VcpkgCommit
}
$CurrentVcpkg = (& git -C $VcpkgRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $CurrentVcpkg -ne $VcpkgCommit) {
    throw "Unexpected vcpkg revision: $CurrentVcpkg; expected $VcpkgCommit"
}

$VcpkgExe = Join-Path $VcpkgRoot "vcpkg.exe"
if (-not (Test-Path $VcpkgExe)) {
    & (Join-Path $VcpkgRoot "bootstrap-vcpkg.bat") -disableMetrics
    if ($LASTEXITCODE -ne 0) {
        throw "vcpkg bootstrap failed with exit code $LASTEXITCODE"
    }
}
if (-not $SkipDependencyInstall) {
    Invoke-Native $VcpkgExe install g2o suitesparse yaml-cpp eigen3 glog opencv sqlite3 --triplet $Triplet
}

$Generator = "Visual Studio 17 2022"
$StellaBuild = Join-Path $StellaRoot "build\msvc-x64-static-release"
$StellaInstall = Join-Path $StellaRoot "install\release"
$Toolchain = Join-Path $VcpkgRoot "scripts\buildsystems\vcpkg.cmake"
Invoke-Native cmake `
    -S $StellaRoot `
    -B $StellaBuild `
    -G $Generator `
    -A x64 `
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain" `
    "-DVCPKG_TARGET_TRIPLET=$Triplet" `
    "-DCMAKE_INSTALL_PREFIX=$StellaInstall" `
    -DBUILD_SHARED_LIBS=OFF `
    -DBUILD_TESTS=OFF `
    -DUSE_GTSAM=OFF `
    -DUSE_OPENMP=OFF `
    -DUSE_SSE_ORB=OFF
Invoke-Native cmake --build $StellaBuild --config Release --target install --parallel

$ExamplesBuild = Join-Path $ExamplesRoot "build"
Invoke-Native cmake `
    -S $ExamplesRoot `
    -B $ExamplesBuild `
    -G $Generator `
    -A x64 `
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain" `
    "-DVCPKG_TARGET_TRIPLET=$Triplet" `
    "-DCMAKE_PREFIX_PATH=$StellaInstall" `
    -DUSE_STACK_TRACE_LOGGER=OFF `
    -DUSE_GOOGLE_PERFTOOLS=OFF
Invoke-Native cmake --build $ExamplesBuild --config Release --target run_video_slam --parallel

$Executable = Join-Path $ExamplesBuild "run_video_slam.exe"
if (-not (Test-Path $Executable)) {
    throw "Missing build output: $Executable"
}
Copy-Item -Force $Executable (Join-Path $RuntimeRoot "run_video_slam.exe")

New-Item -ItemType Directory -Force -Path $VocabRoot | Out-Null
$Vocabulary = Join-Path $VocabRoot "orb_vocab.fbow"
if (-not (Test-Path $Vocabulary)) {
    Invoke-WebRequest -Uri "https://github.com/stella-cv/FBoW_orb_vocab/raw/main/orb_vocab.fbow" -OutFile $Vocabulary
}
$ActualVocabSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Vocabulary).Hash
if ($ActualVocabSha256 -ne $VocabSha256) {
    throw "ORB vocabulary checksum mismatch: $ActualVocabSha256"
}

$Manifest = [ordered]@{
    schema_version = 1
    stella_vslam_commit = $StellaCommit
    stella_vslam_examples_commit = $ExamplesCommit
    vcpkg_commit = $VcpkgCommit
    triplet = $Triplet
    orb_vocab_sha256 = $VocabSha256
    executable = (Join-Path $RuntimeRoot "run_video_slam.exe")
    vocabulary = $Vocabulary
}
$Manifest | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $RuntimeRoot "build_manifest.json")
Write-Host "Built: $(Join-Path $RuntimeRoot 'run_video_slam.exe')"
