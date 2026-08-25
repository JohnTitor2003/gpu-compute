# Nsight Compute on the two kernels that matter for the N<=128 claim.
param(
    [string]$Ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\target\windows-desktop-win7-x64\ncu.exe",
    [string]$Exe = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Exe) { $Exe = Join-Path $root "build\bin\gemm_bench.exe" }
$out = Join-Path $root "results"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;" + $env:PATH

$sections = @(
    "--section", "LaunchStats",
    "--section", "Occupancy",
    "--section", "SpeedOfLight",
    "--section", "MemoryWorkloadAnalysis",
    "--section", "ComputeWorkloadAnalysis",
    "--section", "SpeedOfLight_RooflineChart",
    "--section", "SpeedOfLight_HierarchicalSingleRooflineChart"
)

function Profile([string]$name, [string[]]$args) {
    $rep = Join-Path $out "ncu_$name"
    Write-Host "ncu $name" -ForegroundColor Cyan
    & $Ncu -o $rep -f --csv @sections --target-processes all $Exe @args
    if ($LASTEXITCODE -ne 0) { Write-Warning "ncu $name exit $LASTEXITCODE" }
}

# Square 128: the headline kernel (32x32 tiled via auto).
Profile "square128_auto" @("--m","128","--n","128","--k","128","--only","auto","--warmup","3","--iters","8")
Profile "square4096_auto" @("--m","4096","--n","4096","--k","4096","--only","auto","--warmup","2","--iters","4")
Profile "square4096_wmma" @("--m","4096","--n","4096","--k","4096","--only","wmma","--warmup","2","--iters","4")
Write-Host "reports in $out\ncu_*.ncu-rep / .csv" -ForegroundColor Green
