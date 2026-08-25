#pragma once
// Copyright (c) 2026 JohnTitor2003. MIT. See LICENSE.

#include <cuda_runtime.h>

// Row-major SGEMM: C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N]
//
// Tile vocabulary
//   BM, BN  CTA output tile (rows of C, cols of C)
//   BK      K-step staged through shared memory
//   TM, TN  per-thread microtile kept in registers
//
// A 128x128x8 / 8x8 config is 256 threads, 64 FP32 accumulators/thread
// (64 registers for C alone), ~8.5 KB smem, target occupancy 3–4 CTAs/SM.

__host__ __device__ constexpr int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C) {
  const int row = int(blockIdx.y * blockDim.y + threadIdx.y);
  const int col = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= M || col >= N)
    return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k)
    acc += A[row * K + k] * B[k * N + col];
  C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

inline void launch_naive(int M, int N, int K, float alpha, const float* A,
                         const float* B, float beta, float* C,
                         cudaStream_t s = nullptr) {
  dim3 block(16, 16);
  dim3 grid(ceil_div(N, 16), ceil_div(M, 16));
  sgemm_naive<<<grid, block, 0, s>>>(M, N, K, alpha, A, B, beta, C);
}

template <int TILE>
__global__ void sgemm_tiled(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int row = int(blockIdx.y) * TILE + int(threadIdx.y);
  const int col = int(blockIdx.x) * TILE + int(threadIdx.x);

  float acc = 0.f;
  const int tiles = ceil_div(K, TILE);
  for (int t = 0; t < tiles; ++t) {
    const int a_col = t * TILE + int(threadIdx.x);
    const int b_row = t * TILE + int(threadIdx.y);
    As[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
    Bs[threadIdx.y][threadIdx.x] =
        (b_row < K && col < N) ? B[b_row * N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k)
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N)
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

inline void launch_tiled(int M, int N, int K, float alpha, const float* A,
                         const float* B, float beta, float* C,
                         cudaStream_t s = nullptr) {
  constexpr int TILE = 32;
  dim3 block(TILE, TILE);
  dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
  sgemm_tiled<TILE><<<grid, block, 0, s>>>(M, N, K, alpha, A, B, beta, C);
}

// 2D register-blocked SGEMM. SplitK=true walks K with gridDim.z and
// atomicAdds into C so a skinny or small problem still fills the SMs.
template <int BM, int BN, int BK, int TM, int TN, bool SplitK>
__global__ void sgemm_vec(int M, int N, int K, float alpha,
                          const float* __restrict__ A, const float* __restrict__ B,
                          float beta, float* __restrict__ C) {
  constexpr int nthreads = (BM * BN) / (TM * TN);
  constexpr int BNP = BN + 8;

  const int tid = int(threadIdx.x);
  const int thread_row = tid / (BN / TN);
  const int thread_col = tid % (BN / TN);
  const int c_row = int(blockIdx.y);
  const int c_col = int(blockIdx.x);

  __shared__ float As[BK * BM];
  __shared__ float Bs[BK * BNP];

  const int a_vec = BK / 4;
  const int b_vec = BN / 4;
  const int inner_row_a = tid / a_vec;
  const int inner_col_a = tid % a_vec;
  const int inner_row_b = tid / b_vec;
  const int inner_col_b = tid % b_vec;
  const int a_stride = nthreads / a_vec;
  const int b_stride = nthreads / b_vec;

  const bool fast = ((M % BM) == 0) && ((N % BN) == 0) && ((K % BK) == 0);

  float acc[TM * TN];
#pragma unroll
  for (int i = 0; i < TM * TN; ++i)
    acc[i] = 0.f;

  const float* A_block = A + c_row * BM * K;
  const float* B_block = B + c_col * BN;
  float* C_block = C + c_row * BM * N + c_col * BN;

  const int ktiles = ceil_div(K, BK);
  const int t0 = SplitK ? int(blockIdx.z) : 0;
  const int tstep = SplitK ? int(gridDim.z) : 1;

  for (int t = t0; t < ktiles; t += tstep) {
    const int k0 = t * BK;

    for (int r = inner_row_a; r < BM; r += a_stride) {
      const int grow = c_row * BM + r;
      const int gcol = k0 + inner_col_a * 4;
      float4 v{0.f, 0.f, 0.f, 0.f};
      if (fast || (grow < M && gcol + 3 < K)) {
        v = *reinterpret_cast<const float4*>(&A_block[r * K + gcol]);
      } else if (grow < M) {
        const float* p = A_block + r * K;
        v.x = gcol < K ? p[gcol] : 0.f;
        v.y = gcol + 1 < K ? p[gcol + 1] : 0.f;
        v.z = gcol + 2 < K ? p[gcol + 2] : 0.f;
        v.w = gcol + 3 < K ? p[gcol + 3] : 0.f;
      }
      As[(inner_col_a * 4 + 0) * BM + r] = v.x;
      As[(inner_col_a * 4 + 1) * BM + r] = v.y;
      As[(inner_col_a * 4 + 2) * BM + r] = v.z;
      As[(inner_col_a * 4 + 3) * BM + r] = v.w;
    }

    for (int r = inner_row_b; r < BK; r += b_stride) {
      const int grow = k0 + r;
      const int gcol = c_col * BN + inner_col_b * 4;
      float4 v{0.f, 0.f, 0.f, 0.f};
      if (fast || (grow < K && gcol + 3 < N)) {
        v = *reinterpret_cast<const float4*>(
            &B_block[grow * N + inner_col_b * 4]);
      } else if (grow < K) {
        const float* p = B_block + grow * N;
        v.x = gcol < N ? p[inner_col_b * 4 + 0] : 0.f;
        v.y = gcol + 1 < N ? p[inner_col_b * 4 + 1] : 0.f;
        v.z = gcol + 2 < N ? p[inner_col_b * 4 + 2] : 0.f;
        v.w = gcol + 3 < N ? p[inner_col_b * 4 + 3] : 0.f;
      }
      Bs[r * BNP + inner_col_b * 4 + 0] = v.x;
      Bs[r * BNP + inner_col_b * 4 + 1] = v.y;
      Bs[r * BNP + inner_col_b * 4 + 2] = v.z;
      Bs[r * BNP + inner_col_b * 4 + 3] = v.w;
    }

    __syncthreads();

    float a_reg[TM];
    float b_reg[TN];
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i)
        a_reg[i] = As[k * BM + thread_row * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j)
        b_reg[j] = Bs[k * BNP + thread_col * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i * TN + j] = fmaf(a_reg[i], b_reg[j], acc[i * TN + j]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int row = c_row * BM + thread_row * TM + i;
    if (row >= M)
      continue;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = c_col * BN + thread_col * TN + j;
      if (col >= N)
        continue;
      const int idx = (thread_row * TM + i) * N + (thread_col * TN + j);
      const float v = alpha * acc[i * TN + j];
      if (SplitK)
        atomicAdd(&C_block[idx], v);
      else
        C_block[idx] = v + beta * C_block[idx];
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
inline void launch_vec_cfg(int M, int N, int K, float alpha, const float* A,
                           const float* B, float beta, float* C, int splits,
                           cudaStream_t s) {
  constexpr int threads = (BM * BN) / (TM * TN);
  static_assert(threads >= 32, "CTA too small");
  if (splits <= 1) {
    dim3 block(threads);
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    sgemm_vec<BM, BN, BK, TM, TN, false>
        <<<grid, block, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    if (beta == 0.f)
      cudaMemsetAsync(C, 0, size_t(M) * size_t(N) * sizeof(float), s);
    dim3 block(threads);
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM), splits);
    sgemm_vec<BM, BN, BK, TM, TN, true>
        <<<grid, block, 0, s>>>(M, N, K, alpha, A, B, 0.f, C);
  }
}

inline void launch_vec(int M, int N, int K, float alpha, const float* A,
                       const float* B, float beta, float* C,
                       cudaStream_t s = nullptr) {
  if (M <= 64 && N <= 64)
    launch_vec_cfg<64, 64, 16, 4, 4>(M, N, K, alpha, A, B, beta, C, 1, s);
  else
    launch_vec_cfg<128, 128, 8, 8, 8>(M, N, K, alpha, A, B, beta, C, 1, s);
}

// Skinny N: B is reused by every warp in the CTA (it does not depend on
// row). Stage a 32 x N slab in smem once per K-step; each warp owns one
// output row; lanes walk K with stride 32 so A is fully coalesced; then
// shuffle-reduce the N accumulators. Compile-time Nmax unrolls the inner
// FMA and keeps C in registers — 1 float/thread for N=1, 32 for N=32.
template <int Nmax>
__global__ void sgemm_skinny_warp(int M, int N, int K, float alpha,
                                  const float* __restrict__ A,
                                  const float* __restrict__ B, float beta,
                                  float* __restrict__ C) {
  const int lane = int(threadIdx.x) & 31;
  const int warp = int(threadIdx.x) >> 5;
  const int warps = int(blockDim.x) >> 5;
  const int row = int(blockIdx.x) * warps + warp;
  if (row >= M)
    return;

  float acc[Nmax];
#pragma unroll
  for (int n = 0; n < Nmax; ++n)
    acc[n] = 0.f;

  for (int k = lane; k < K; k += 32) {
    const float a = A[row * K + k];
#pragma unroll
    for (int n = 0; n < Nmax; ++n) {
      if (n < N)
        acc[n] = fmaf(a, B[k * N + n], acc[n]);
    }
  }

#pragma unroll
  for (int n = 0; n < Nmax; ++n) {
    if (n >= N)
      break;
    float v = acc[n];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0)
      C[row * N + n] = alpha * v + beta * C[row * N + n];
  }
}

inline void launch_skinny(int M, int N, int K, float alpha, const float* A,
                          const float* B, float beta, float* C,
                          cudaStream_t s = nullptr) {
  constexpr int threads = 256;  // 8 warps = 8 rows / CTA
  const int grid = ceil_div(M, threads / 32);

#define LAUNCH_SKINNY(NV)                                                      \
  sgemm_skinny_warp<NV>                                                        \
      <<<grid, threads, 0, s>>>(M, N, K, alpha, A, B, beta, C)

  switch (N) {
    case 1:
      LAUNCH_SKINNY(1);
      break;
    case 2:
      LAUNCH_SKINNY(2);
      break;
    case 4:
      LAUNCH_SKINNY(4);
      break;
    case 8:
      LAUNCH_SKINNY(8);
      break;
    case 16:
      LAUNCH_SKINNY(16);
      break;
    default:
      LAUNCH_SKINNY(32);
      break;
  }
#undef LAUNCH_SKINNY
}

inline int occupancy_splits(int M, int N, int BM, int BN) {
  const int tiles = ceil_div(M, BM) * ceil_div(N, BN);
  // Blackwell 5080 has 84 SMs; want ≥4 CTAs/SM in flight.
  const int want = 84 * 4;
  int splits = ceil_div(want, tiles < 1 ? 1 : tiles);
  if (splits < 1)
    splits = 1;
  if (splits > 16)
    splits = 16;
  return splits;
}

inline void launch_auto(int M, int N, int K, float alpha, const float* A,
                        const float* B, float beta, float* C,
                        cudaStream_t s = nullptr) {
  // Rank-1..32: warp-per-row, smem-resident B, coalesced A.
  // N=64/128 tall-skinny: 64-row register tiles + split-K so the grid
  // fills 84 SMs instead of launching a handful of fat CTAs.
  // Small square: 32x32 tiling (enough CTAs that the 128x128 kernel
  // would underfill). Large square: 128x128 / 8x8 register blocking.
  if (N <= 32 && N > 0 && M >= 256 && K >= 256) {
    launch_skinny(M, N, K, alpha, A, B, beta, C, s);
  } else if (N <= 64 && M >= 512 && K >= 512) {
    const int splits = occupancy_splits(M, N, 64, 64);
    launch_vec_cfg<64, 64, 16, 4, 4>(M, N, K, alpha, A, B, beta, C, splits, s);
  } else if (N <= 128 && M >= 512 && K >= 512) {
    const int splits = occupancy_splits(M, N, 64, 128);
    launch_vec_cfg<64, 128, 8, 4, 8>(M, N, K, alpha, A, B, beta, C, splits, s);
  } else if (M <= 64 && N <= 64) {
    launch_vec(M, N, K, alpha, A, B, beta, C, s);
  } else if (M <= 384 && N <= 384) {
    launch_tiled(M, N, K, alpha, A, B, beta, C, s);
  } else {
    launch_vec(M, N, K, alpha, A, B, beta, C, s);
  }
}
