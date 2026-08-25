# Run the three kernels and dump logs under results/.
param(
    [string]$BuildDir = "build",
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$bin = Join-Path $root "$BuildDir\bin"
if (-not (Test-Path $bin)) { $bin = Join-Path $root "$BuildDir\bin\Release" }
if (-not (Test-Path $bin)) { throw "Build first: cmake --build $BuildDir --config Release" }

New-Item -ItemType Directory -Force -Path (Join-Path $root "results") | Out-Null

$gemm = Join-Path $bin "gemm_bench.exe"
$pt = Join-Path $bin "pathtracer.exe"
$nb = Join-Path $bin "nbody.exe"

Write-Host "== SGEMM vs cuBLAS ==" -ForegroundColor Cyan
$gemmArgs = @()
if ($Quick) { $gemmArgs = @("--quick") }
& $gemm @gemmArgs | Tee-Object (Join-Path $root "results\gemm.txt")

$spp = 256
$nbodyN = 32768
$nbodySteps = 96
if ($Quick) { $spp = 32; $nbodyN = 8192; $nbodySteps = 32 }

Write-Host "== Path tracer ==" -ForegroundColor Cyan
& $pt --width 800 --height 800 --spp $spp --out results/cornell.png |
    Tee-Object (Join-Path $root "results\pathtracer.txt")

Write-Host "== N-body ==" -ForegroundColor Cyan
& $nb --n $nbodyN --steps $nbodySteps --out results/nbody.png |
    Tee-Object (Join-Path $root "results\nbody.txt")

Write-Host "Logs and images in results/" -ForegroundColor Green
