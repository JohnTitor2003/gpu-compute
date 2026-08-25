#pragma once

#include <cuda_runtime.h>

// All-pairs gravitational N-body.
// Position/mass packed as float4 (x,y,z,mass). Velocity as float4 (x,y,z,_).

constexpr int kNBodyTile = 128;

__device__ __forceinline__ float3 acc_from(float4 pi, float4 pj, float eps2) {
  const float dx = pj.x - pi.x;
  const float dy = pj.y - pi.y;
  const float dz = pj.z - pi.z;
  const float r2 = dx * dx + dy * dy + dz * dz + eps2;
  const float inv = rsqrtf(r2);
  const float inv3 = inv * inv * inv;
  const float s = pj.w * inv3;
  return make_float3(dx * s, dy * s, dz * s);
}

__global__ void nbody_naive(const float4* __restrict__ pos, float4* acc, int n,
                            float eps2) {
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n)
    return;
  const float4 pi = pos[i];
  float ax = 0.f, ay = 0.f, az = 0.f;
  for (int j = 0; j < n; ++j) {
    const float3 a = acc_from(pi, pos[j], eps2);
    ax += a.x;
    ay += a.y;
    az += a.z;
  }
  acc[i] = make_float4(ax, ay, az, 0.f);
}

// Tiled: each CTA walks the source particles in kNBodyTile-wide chunks,
// staging them in shared memory so each GMEM load of a source body is reused
// by every thread in the CTA. This is the O(N^2) algorithm, just bandwidth-optimal.
template <int TILE>
__global__ void nbody_tiled(const float4* __restrict__ pos, float4* acc, int n,
                            float eps2) {
  __shared__ float4 tile[TILE];
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  const float4 pi = (i < n) ? pos[i] : make_float4(0, 0, 0, 0);
  float ax = 0.f, ay = 0.f, az = 0.f;

  const int ntiles = (n + TILE - 1) / TILE;
  for (int t = 0; t < ntiles; ++t) {
    const int src = t * TILE + int(threadIdx.x);
    tile[threadIdx.x] =
        (src < n) ? pos[src] : make_float4(0.f, 0.f, 0.f, 0.f);
    __syncthreads();
#pragma unroll 8
    for (int j = 0; j < TILE; ++j) {
      const float3 a = acc_from(pi, tile[j], eps2);
      ax += a.x;
      ay += a.y;
      az += a.z;
    }
    __syncthreads();
  }
  if (i < n)
    acc[i] = make_float4(ax, ay, az, 0.f);
}

__global__ void nbody_integrate(float4* pos, float4* vel, const float4* acc,
                                int n, float dt) {
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n)
    return;
  float4 v = vel[i];
  const float4 a = acc[i];
  float4 p = pos[i];
  v.x += a.x * dt;
  v.y += a.y * dt;
  v.z += a.z * dt;
  p.x += v.x * dt;
  p.y += v.y * dt;
  p.z += v.z * dt;
  vel[i] = v;
  pos[i] = p;
}

// Kinetic + approximate potential (pair sum, GPU reduction-friendly: each
// thread accumulates its half-pairs). Softened Newtonian potential.
__global__ void nbody_energy(const float4* pos, const float4* vel, int n,
                             float eps2, float* out_ke, float* out_pe) {
  __shared__ float ke_s[256];
  __shared__ float pe_s[256];
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  float ke = 0.f, pe = 0.f;
  if (i < n) {
    const float4 p = pos[i];
    const float4 v = vel[i];
    ke = 0.5f * p.w * (v.x * v.x + v.y * v.y + v.z * v.z);
    for (int j = i + 1; j < n; ++j) {
      const float4 q = pos[j];
      const float dx = q.x - p.x;
      const float dy = q.y - p.y;
      const float dz = q.z - p.z;
      const float r = sqrtf(dx * dx + dy * dy + dz * dz + eps2);
      pe -= p.w * q.w / r;
    }
  }
  ke_s[threadIdx.x] = ke;
  pe_s[threadIdx.x] = pe;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      ke_s[threadIdx.x] += ke_s[threadIdx.x + s];
      pe_s[threadIdx.x] += pe_s[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(out_ke, ke_s[0]);
    atomicAdd(out_pe, pe_s[0]);
  }
}

__global__ void nbody_splat(const float4* pos, int n, int w, int h, float scale,
                            unsigned int* bins) {
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n)
    return;
  const float4 p = pos[i];
  const int x = int((p.x * scale + 0.5f) * float(w));
  const int y = int((0.5f - p.y * scale) * float(h));
  if (x >= 0 && x < w && y >= 0 && y < h)
    atomicAdd(&bins[y * w + x], 1u);
}

inline void launch_naive(const float4* pos, float4* acc, int n, float eps2,
                         cudaStream_t s = nullptr) {
  nbody_naive<<<(n + 255) / 256, 256, 0, s>>>(pos, acc, n, eps2);
}

inline void launch_tiled(const float4* pos, float4* acc, int n, float eps2,
                         cudaStream_t s = nullptr) {
  nbody_tiled<kNBodyTile>
      <<<(n + kNBodyTile - 1) / kNBodyTile, kNBodyTile, 0, s>>>(pos, acc, n,
                                                                eps2);
}
