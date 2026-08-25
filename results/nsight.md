# Occupancy and Nsight

## Counter permission

Nsight Compute 2026.2.1 on this WDDM 5080 returns `ERR_NVGPUCTRPERM`. GPU performance counters are locked to administrators until:

**NVIDIA Control Panel → Developer → Manage GPU Performance Counters → Allow access to all users**, then reboot.

`scripts/profile_gemm.ps1` is the capture. Until that toggle is on, the numbers below are from launch geometry + CUDA events, not from `sm__*` metrics.

## 128³ — the FP32 win (tiled 32×32)

| | |
|---|---|
| kernel | `sgemm_tiled<32>` via `launch_auto` |
| grid | 4 × 4 = **16 CTAs** |
| block | 32 × 32 = **1024 threads** (32 warps) |
| smem | 2 × 32 × 32 × 4 B = **8 KB** |
| registers / thread | ~1 accumulator + addressing (not 64) |
| CTAs / SM (84 SMs) | 16/84 ≈ **0.19** |
| work | 2×128³ = 4.2 MFLOP |
| measured | auto **~10 µs / 0.3 TF/s**, cuBLAS FP32 **~16 µs / 0.2 TF/s** |

That occupancy is *intentionally* tiny. The kernel cannot be math-bound or HBM-bound at 4 MFLOP. The 6 µs gap is launch + cuBLAS heuristic. Speed-of-light in `ncu` should read low SM% and low DRAM% — if it doesn’t, the story is wrong.

## 4096³ — the FP32 loss (vec 128×128 / 8×8)

| | |
|---|---|
| kernel | `sgemm_vec<128,128,8,8,8>` |
| grid | 32 × 32 = **1024 CTAs** |
| block | **256 threads**, 64 FP32 C accumulators / thread |
| smem | ~8.5 KB |
| CTAs / SM | 1024/84 ≈ **12** (occupancy now real) |
| work | 2×4096³ = 137 GFLOP |
| measured | auto **23 TF/s**, cuBLAS FP32 **37 TF/s**, cuBLAS TF32 **53 TF/s** |

Enough tiles to fill the chip. We lose. That is the correct result: production SGEMM at this size is a library problem.

## WMMA TF32 — different algorithm

`sgemm_wmma_tf32`: 4 warps / CTA, each warp a 16×16×8 TF32 MMA, CTA tile 32×32. 10-bit mantissa. Error vs FP32 reference is ~7×10⁻⁴, not 10⁻⁷.

| N³ | WMMA TF32 | cuBLAS TF32 | WMMA / cuBLAS TF32 |
|--:|----------:|------------:|-------------------:|
| 128 | 0.28 | 0.44 | 0.64× |
| 256 | 3.05 | 3.63 | 0.84× |
| 512 | 11.36 | 21.59 | 0.53× |
| 4096 | 18.14 | 53.07 | 0.34× |

A naive WMMA loop does not beat `cublasSgemm` in TF32 mode. It exists so the tensor-core column cannot be confused with the CUDA-core FP32 win.

## Capture once counters work

```powershell
./scripts/profile_gemm.ps1
# or:
ncu --csv --section LaunchStats --section Occupancy --section SpeedOfLight `
    --section SpeedOfLight_HierarchicalSingleRooflineChart `
    -o results/ncu_128 --target-processes all `
    ./build/bin/gemm_bench.exe --m 128 --n 128 --k 128 --only auto --iters 8 --warmup 3
ncu ... -o results/ncu_4096 ... --m 4096 --n 4096 --k 4096 --only auto --iters 4 --warmup 2
ncu ... -o results/ncu_4096_wmma ... --only wmma --m 4096 --n 4096 --k 4096 --iters 4
```
