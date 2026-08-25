# SGEMM kernel — tiling, registers, Blackwell

Hand-written row-major `C = A @ B`. Compared to `cublasSgemm` in `CUBLAS_PEDANTIC_MATH` on an RTX 5080 (sm_120).

**Claim, measured:** faster than cuBLAS FP32 on every square `N ∈ {32,48,64,80,96,112,128,160,192,256}`. Not faster on tall-skinny `N ≤ 128` with `M=K=4096` — cuBLAS GEMV is already on the HBM roof.

![square speedup](../results/speedup_square.png)

## Tiling

Three levels, only two of which fire on the win regime.

```
                    N (columns of C)
              ┌─────────────────────┐
            M │  CTA tile  BM × BN  │     BM = BN = 32  (win kernel)
              │  ┌─────┐            │     BM = BN = 128 (large vec)
              │  │TM×TN│ thread     │     TM = TN = 8   → 64 C registers
              │  └─────┘            │
              └─────────────────────┘
                         K walked in steps of BK
                         (32 for tiled, 8 for vec)
```

**32×32 tiled (the kernel that beats cuBLAS at N≤256)**

- Grid: `(N/32) × (M/32)` CTAs. At 32³ that is **1 CTA**. At 128³, **16 CTAs**. At 256³, **64 CTAs**.
- Block: 32×32 threads = 1024 threads = 32 warps.
- Each thread owns **one** `C[i,j]`. Inner product over `K` in steps of 32.
- Shared memory: `As[32][32] + Bs[32][32]` = 8 KB. Double-buffered would cost occupancy we do not have at 1–16 CTAs; we do not bother.
- GMEM loads are coalesced: thread `(tx,ty)` reads `A[row, k0+tx]` and `B[k0+ty, col]` — consecutive `tx` walk consecutive columns of A.

Why this wins: cuBLAS still runs its general GEMM chooser. That chooser is correct for 4096³ and heavy for 32³. Our launch is one (or sixteen) specialized CTAs with a fully unrolled 32-long inner product. The 16.2 µs cuBLAS floor on this machine is almost independent of `N` from 32 to 128; our time is 8–11 µs. The gap *is* the heuristic plus a too-fat tile.

**128×128 / 8×8 vec (large square)**

- 256 threads. Each thread owns an 8×8 microtile → **64 FP32 accumulators in registers**.
- `BK=8`. A-tile stored **transposed** in smem (`BK × BM`) so the `k` loop streams consecutive 32-bit words (no 32-way bank conflict on A).
- B-tile padded to `BN+8` so the 8-wide `TN` loads are not 4-way bank-conflicted (`threadCol*8` would otherwise alias every 4 threads onto the same 32 banks).
- Vectorized `float4` GMEM loads. Fast path skips bounds checks when `M,N,K` divide the tile.
- Occupancy limit: 64 C registers + `a_reg[8]` + `b_reg[8]` ≈ 80+ registers. We do **not** `__launch_bounds__(256, 3)` this; spilling would lose to the 32×32 kernel on the sizes that matter.

**Skinny warp (N ≤ 32, large M,K)**

- 8 warps / CTA, **one output row per warp**.
- Lanes walk `K` with stride 32 → A loads are 128-byte coalesced.
- `N` accumulators in registers (compile-time `Nmax`), warp-shuffle reduce, lane 0 writes the row of C.
- `B` is `N×K` and L2-resident. For `N=1` this is GEMV: traffic is a 64 MB A read. cuBLAS hits ~940 GB/s of 960 GB/s HBM. There is no kernel trick that beats a library already on the roofline. We keep the kernel because the *code* is the right shape; we do not put it in the win column.

**Split-K vec (N = 64 or 128, tall M)**

- A 64×N register tile with `gridDim.z` K-splits so 84 SMs see ≥4 CTAs.
- Partials `atomicAdd` into a zeroed C. 4096×64×4096 lands at **0.90×** cuBLAS — close, not a win.

## Register budget (win kernel)

| item | 32×32 tiled | 128×128 / 8×8 vec |
|---|---|---|
| C accumulators | 1 | 64 |
| A/B staging | smem 8 KB | smem ~8.5 KB + 8+8 regs |
| occupancy story | 1024 thr/CTA, few CTAs, **launch-bound** | 256 thr/CTA, register-heavy, needs many tiles |
| chosen for N≤256? | **yes** | no |

Blackwell (GB203) has 64K 32-bit registers/SM and 84 SMs. The 32×32 kernel barely touches that. The 8×8 kernel would, which is why `launch_auto` refuses to use it until `M,N > 384`.

## Why Blackwell specifically

1. **84 SMs, fat launch path.** Kernel-selection in `cublasSgemm` is a larger fraction of a 16 µs launch than it was on a 20-SM laptop GPU. Size-specialized entry points pick up more, not less.
2. **sm_120 + CUDA 13.3.** `cp.async` / TMA exist; we do not use them on the win kernel. A 32×32 tile is too small for TMA to pay. Using Hopper machinery here would be résumé-driven design. We didn’t.
3. **HBM at 960 GB/s.** Skinny GEMM is a bandwidth problem. The library already solved it. The interesting leftover is the launch-bound square.

## Reproducing

```powershell
./scripts/build.ps1
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64;" + $env:PATH
./build/bin/gemm_bench.exe --suite win --csv results/gemm.csv
python scripts/plot_gemm.py
```

| flag | |
|---|---|
| `--suite win` | square 32–256 and skinny N≤128 (default) |
| `--suite full` | up to 4096³ |
| `--csv path` | machine-readable |
| `--m --n --k` | one shape (for Nsight) |
| `--iters N` | pin iteration count |
| `--skip-ladder` | auto + cuBLAS only |

Nsight: `./scripts/profile_gemm.ps1`. If you see `ERR_NVGPUCTRPERM`, enable counters in the NVIDIA Control Panel (Developer → Manage GPU Performance Counters) or run elevated. The software roofline in `results/roofline.png` does not need that.

## What we are not claiming

- Beating TF32 / tensor-core `cublasGemmEx` with FP32 CUDA cores.
- Beating cuBLAS on 1024³ and up. `4096³` loses, as it should.
- Beating cuBLAS on `N=1..128` skinny with large `M,K`. Measured; red chart.
