# Nsight notes

`scripts/profile_gemm.ps1` runs Nsight Compute 2026.2.1 on `--m 128 --n 128 --k 128 --skip-ladder`.

On this machine the capture failed with `ERR_NVGPUCTRPERM`: the Windows account does not have GPU performance-counter access. Enable it in **NVIDIA Control Panel → Developer → Manage GPU Performance Counters → Allow access to the GPU performance counters to all users**, reboot, re-run. Elevated shells sometimes skip the reboot.

Until counters are on, the software roofline (`roofline.png`) is the profile: FLOP/s from CUDA events against `2MNK / 4(MK+KN+MN)` bytes. Small squares sit well below the 59 FLOP/byte ridge — launch-bound, which is the win condition we actually use.

What to look for once `ncu` works:

| section | 32×32 tiled at 128³ (16 CTAs) |
|---|---|
| LaunchStats | 1024 threads/CTA, 16 CTAs, smem 8 KB |
| Occupancy | limited by threads/CTA (1024), not registers |
| SpeedOfLight | low SM% and DRAM% — confirms launch-bound, not math-bound |
| Roofline (single) | left of ridge, below HBM roof |
