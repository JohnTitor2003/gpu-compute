# Tiled N-body

All-pairs gravitational N-body on the GPU. The algorithm is O(N²); the kernel is the NVIDIA “tile in shared memory” formulation, which is the bandwidth-optimal way to evaluate that algorithm.

## Kernel

Each CTA owns `TILE=128` target bodies. It walks the source bodies in 128-wide chunks, stages a chunk in shared memory, and lets every thread in the CTA reuse those 128 loads. One GMEM load of a source `float4` therefore feeds 128 inner-product-style accumulations instead of one.

Force accounting is 20 FLOP per pair (3 sub, r², rsqrt, inv³, 3 scaled adds). Softening `ε²` is added to `r²` so the merger does not NaN when two bodies meet.

Integration is symplectic Euler (kick-drift) with a shared `dt`. Energy (kinetic + softened potential) is reduced on device before and after the step loop so drift is a number, not a vibe.

## Initial condition

Two Plummer spheres, offset on x, with opposing bulk velocity. The snapshot is a 2D histogram of the projection, log-scaled, viridis-colored.

## Build and run

```bash
cmake --build build --config Release --target nbody
./build/bin/nbody --n 32768 --steps 96 --out results/nbody.png
```

| flag | default | |
|---|---|---|
| `--n` | 16384 | must be a multiple of 128 |
| `--steps` | 64 | |
| `--dt` | 0.002 | |
| `--eps` | 0.02 | softening length |
| `--skip-naive` | off | skip the O(N²) global-memory baseline (recommended for N>32k) |
| `--out` | `results/nbody.png` | |

The bench prints tiled vs naive milliseconds, interactions/s, TFLOP/s, and relative energy drift. Logs from this machine: [`../results/nbody.txt`](../results/nbody.txt), image [`../results/nbody.png`](../results/nbody.png).
