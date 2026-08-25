# Build the three CUDA binaries with nvcc + MSVC, no CMake required.
param(
    [string]$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
    [string]$VsVcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    [string]$Arch = "sm_120",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = Join-Path $root "build\bin" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if (-not (Test-Path $VsVcvars)) { throw "vcvars64.bat not found: $VsVcvars" }
if (-not (Test-Path (Join-Path $CudaRoot "bin\nvcc.exe"))) { throw "nvcc not found under $CudaRoot" }

$common = "-O3 --use_fast_math -std=c++17 -lineinfo --expt-relaxed-constexpr -arch=$Arch -I`"$root`" --compiler-options /O2,/std:c++17,/EHsc,/wd4819"
$targets = @(
    @{ Name = "gemm_bench"; Src = "gemm\bench_gemm.cu"; Libs = "-lcublas" },
    @{ Name = "pathtracer"; Src = "pathtracer\main.cu"; Libs = "" },
    @{ Name = "nbody";      Src = "nbody\main.cu";      Libs = "" }
)

$nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
foreach ($t in $targets) {
    $out = Join-Path $OutDir ($t.Name + ".exe")
    $src = Join-Path $root $t.Src
    Write-Host "nvcc $($t.Name) -> $out" -ForegroundColor Cyan
    $cmd = "`"$VsVcvars`" && `"$nvcc`" $common $($t.Libs) `"$src`" -o `"$out`""
    & cmd.exe /c $cmd
    if ($LASTEXITCODE -ne 0) { throw "nvcc failed for $($t.Name) (exit $LASTEXITCODE)" }
}
Write-Host "Built:" -ForegroundColor Green
Get-ChildItem $OutDir -Filter *.exe | ForEach-Object { Write-Host "  $($_.FullName)" }
Write-Host "To run gemm_bench, put CUDA DLLs on PATH:" -ForegroundColor Yellow
Write-Host "  `$env:PATH = '$CudaRoot\bin\x64;' + `$env:PATH"
