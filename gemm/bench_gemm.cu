#include "common/cuda.hpp"
#include "gemm/kernels.cuh"

#include <cublas_v2.h>

#define CUBLAS_CHECK(expr)                                                     \
  do {                                                                         \
    cublasStatus_t st__ = (expr);                                              \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "cuBLAS error %s:%d: %s = %d\n", __FILE__,          \
                   __LINE__, #expr, static_cast<int>(st__));                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void fill_host(std::vector<float>& x, unsigned seed) {
  unsigned s = seed;
  for (size_t i = 0; i < x.size(); ++i) {
    s = s * 1664525u + 1013904223u;
    x[i] = float(int(s >> 9) & 0xFFFF) / 65535.f * 2.f - 1.f;
  }
}

static void cublas_sgemm_rowmajor(cublasHandle_t h, int M, int N, int K,
                                  float alpha, const float* A, const float* B,
                                  float beta, float* C) {
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N,
                           A, K, &beta, C, N));
}

using LaunchFn = void (*)(int, int, int, float, const float*, const float*,
                          float, float*, cudaStream_t);

static float bench_ms(LaunchFn fn, int M, int N, int K, float alpha,
                      const float* A, const float* B, float beta, float* C,
                      int warmup, int iters) {
  for (int i = 0; i < warmup; ++i)
    fn(M, N, K, alpha, A, B, beta, C, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuTimer t;
  t.record_start();
  for (int i = 0; i < iters; ++i)
    fn(M, N, K, alpha, A, B, beta, C, nullptr);
  t.record_stop();
  CUDA_CHECK(cudaGetLastError());
  return t.elapsed_ms() / float(iters);
}

static double flop(int M, int N, int K) {
  return 2.0 * double(M) * double(N) * double(K);
}
static double bytes_moved(int M, int N, int K) {
  return 4.0 * (double(M) * K + double(K) * N + double(M) * N);
}
static double tflops(int M, int N, int K, float ms) {
  return flop(M, N, K) / (double(ms) * 1.0e9);
}
static double ai(int M, int N, int K) {
  return flop(M, N, K) / bytes_moved(M, N, K);
}

static float max_rel_err(const std::vector<float>& ref,
                         const std::vector<float>& got) {
  double max_ref = 0.0;
  for (float v : ref)
    max_ref = std::max(max_ref, std::abs(double(v)));
  double worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i)
    worst = std::max(worst, std::abs(double(ref[i]) - double(got[i])));
  return float(worst / (max_ref + 1e-8));
}

struct Row {
  std::string label;
  int M, N, K;
  float naive, tiled, vec, ours, cublas_fp32, cublas_tf32, err;
  bool beat;
  bool have_naive, have_tiled, have_vec;
};

static void print_table(const char* title, const std::vector<Row>& rows) {
  std::printf("\n## %s\n\n", title);
  std::printf("| shape | AI | auto µs | cuBLAS FP32 µs | speedup | auto TF/s | "
              "cuBLAS FP32 TF/s | cuBLAS TF32 TF/s | tiled TF/s | vec TF/s | "
              "result | err |\n");
  std::printf("|------:|---:|--------:|---------------:|--------:|----------:|"
              "-----------------:|-----------------:|-----------:|---------:|"
              "-------:|----:|\n");
  for (const auto& r : rows) {
    const float sp = r.ours > 0.f ? r.cublas_fp32 / r.ours : 0.f;
    char tiled_s[16], vec_s[16];
    if (r.have_tiled)
      std::snprintf(tiled_s, sizeof(tiled_s), "%.2f",
                    tflops(r.M, r.N, r.K, r.tiled));
    else
      std::snprintf(tiled_s, sizeof(tiled_s), "—");
    if (r.have_vec)
      std::snprintf(vec_s, sizeof(vec_s), "%.2f", tflops(r.M, r.N, r.K, r.vec));
    else
      std::snprintf(vec_s, sizeof(vec_s), "—");
    std::printf("| %s | %.1f | %.1f | %.1f | **%.2fx** | %.2f | %.2f | %.2f | "
                "%s | %s | %s | %.1e |\n",
                r.label.c_str(), ai(r.M, r.N, r.K), r.ours * 1e3f,
                r.cublas_fp32 * 1e3f, sp, tflops(r.M, r.N, r.K, r.ours),
                tflops(r.M, r.N, r.K, r.cublas_fp32),
                tflops(r.M, r.N, r.K, r.cublas_tf32), tiled_s, vec_s,
                r.beat ? "WIN" : "lose", r.err);
  }
  std::printf("\nµs is kernel time (CUDA events, warmed). Speedup is "
              "t_cublas_fp32 / t_auto. AI is FLOP/byte against the 2MN+2MK+2KN "
              "traffic lower bound. WIN = faster than pedantic cuBLAS FP32.\n");
}

int main(int argc, char** argv) {
  print_device();

  const float alpha = 1.f, beta = 0.f;
  const int warmup = std::atoi(arg_value(argc, argv, "--warmup", "15").c_str());
  const int iters_cli = std::atoi(arg_value(argc, argv, "--iters", "0").c_str());
  const std::string suite = arg_value(argc, argv, "--suite", "win");
  const std::string csv_path = arg_value(argc, argv, "--csv", "");
  const bool skip_naive = arg_flag(argc, argv, "--skip-naive") || suite == "win";
  const bool skip_ladder = arg_flag(argc, argv, "--skip-ladder");
  const bool do_naive = !skip_naive && !skip_ladder;
  const bool do_ladder = !skip_ladder;

  const int one_m = std::atoi(arg_value(argc, argv, "--m", "0").c_str());
  const int one_n = std::atoi(arg_value(argc, argv, "--n", "0").c_str());
  const int one_k = std::atoi(arg_value(argc, argv, "--k", "0").c_str());

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  auto run_cublas = [&](int M, int N, int K, const float* A, const float* B,
                        float* C, cublasMath_t math, int iters) {
    CUBLAS_CHECK(cublasSetMathMode(handle, math));
    for (int i = 0; i < warmup; ++i)
      cublas_sgemm_rowmajor(handle, M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer t;
    t.record_start();
    for (int i = 0; i < iters; ++i)
      cublas_sgemm_rowmajor(handle, M, N, K, alpha, A, B, beta, C);
    t.record_stop();
    return t.elapsed_ms() / float(iters);
  };

  auto iters_for = [&](int M, int N, int K) {
    const double f = flop(M, N, K);
    int it = int(1.5e11 / f);  // ~150 ms at 1 TFLOP/s
    if (it < 40)
      it = 40;
    if (it > 4000)
      it = 4000;
    return it;
  };

  struct Shape {
    int M, N, K;
  };
  std::vector<Shape> squares, skinny;

  if (one_m > 0 && one_n > 0 && one_k > 0) {
    skinny.push_back({one_m, one_n, one_k});
  } else if (suite == "quick") {
    squares = {{64, 64, 64}, {128, 128, 128}, {256, 256, 256}};
    skinny = {{4096, 4, 4096}, {4096, 16, 4096}};
  } else if (suite == "full") {
    for (int n : {32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096})
      squares.push_back({n, n, n});
    for (int n : {1, 2, 4, 8, 16, 32, 48, 64, 96, 128})
      skinny.push_back({4096, n, 4096});
  } else {
    // --suite win : the claim surface. Small squares and N<=128 skinny.
    for (int n : {32, 48, 64, 80, 96, 112, 128, 160, 192, 256})
      squares.push_back({n, n, n});
    for (int n : {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128})
      skinny.push_back({4096, n, 4096});
    for (int n : {1, 4, 8, 16, 32, 64, 128})
      skinny.push_back({8192, n, 8192});
  }

  FILE* csv = nullptr;
  if (!csv_path.empty()) {
    ensure_parent_dir(csv_path);
#if defined(_MSC_VER)
    fopen_s(&csv, csv_path.c_str(), "w");
#else
    csv = std::fopen(csv_path.c_str(), "w");
#endif
    if (!csv) {
      std::fprintf(stderr, "cannot write %s\n", csv_path.c_str());
      return 1;
    }
    std::fprintf(csv, "family,shape,M,N,K,ai,auto_ms,cublas_fp32_ms,cublas_"
                      "tf32_ms,tiled_ms,vec_ms,naive_ms,auto_tflops,cublas_"
                      "fp32_tflops,cublas_tf32_tflops,speedup_fp32,win,"
                      "relerr,bytes\n");
  }

  std::vector<Row> square_rows, skinny_rows;

  auto run_shape = [&](int M, int N, int K, const char* family,
                       std::vector<Row>& out) {
    std::vector<float> hA(size_t(M) * K), hB(size_t(K) * N),
        hC(size_t(M) * N), hRef(size_t(M) * N);
    fill_host(hA, 1);
    fill_host(hB, 2);

    float* A = device_alloc<float>(hA.size());
    float* B = device_alloc<float>(hB.size());
    float* C = device_alloc<float>(hC.size());
    float* Cref = device_alloc<float>(hC.size());
    CUDA_CHECK(cudaMemcpy(A, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(Cref, 0, hC.size() * 4));

    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    cublas_sgemm_rowmajor(handle, M, N, K, alpha, A, B, beta, Cref);
    CUDA_CHECK(cudaDeviceSynchronize());

    const int it = iters_cli > 0 ? iters_cli : iters_for(M, N, K);
    Row r{};
    r.M = M;
    r.N = N;
    r.K = K;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%d×%d×%d", M, N, K);
    r.label = buf;
    r.have_naive = do_naive && (flop(M, N, K) < 5e10);
    r.have_tiled = do_ladder;
    r.have_vec = do_ladder;

    if (r.have_naive)
      r.naive = bench_ms(launch_naive, M, N, K, alpha, A, B, beta, C, warmup, it);
    if (r.have_tiled)
      r.tiled = bench_ms(launch_tiled, M, N, K, alpha, A, B, beta, C, warmup, it);
    if (r.have_vec)
      r.vec = bench_ms(launch_vec, M, N, K, alpha, A, B, beta, C, warmup, it);
    r.ours = bench_ms(launch_auto, M, N, K, alpha, A, B, beta, C, warmup, it);

    CUDA_CHECK(cudaMemcpy(hC.data(), C, hC.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(hRef.data(), Cref, hRef.size() * 4, cudaMemcpyDeviceToHost));
    r.err = max_rel_err(hRef, hC);

    r.cublas_fp32 = run_cublas(M, N, K, A, B, C, CUBLAS_PEDANTIC_MATH, it);
    r.cublas_tf32 =
        run_cublas(M, N, K, A, B, C, CUBLAS_TF32_TENSOR_OP_MATH, it);
    r.beat = r.ours < r.cublas_fp32;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(Cref);
    out.push_back(r);

    const float sp = r.cublas_fp32 / r.ours;
    std::printf("  %s  auto %7.1f µs  %5.2f TF/s | cublasFP32 %7.1f µs  "
                "%5.2f TF/s | %s **%.2fx**  err %.1e  AI %.1f\n",
                r.label.c_str(), r.ours * 1e3f, tflops(M, N, K, r.ours),
                r.cublas_fp32 * 1e3f, tflops(M, N, K, r.cublas_fp32),
                r.beat ? "WIN" : "lose", sp, r.err, ai(M, N, K));
    std::fflush(stdout);

    if (csv) {
      std::fprintf(csv,
                   "%s,%s,%d,%d,%d,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%."
                   "4f,%.4f,%.4f,%d,%.3e,%.0f\n",
                   family, r.label.c_str(), M, N, K, ai(M, N, K), r.ours,
                   r.cublas_fp32, r.cublas_tf32, r.have_tiled ? r.tiled : -1,
                   r.have_vec ? r.vec : -1, r.have_naive ? r.naive : -1,
                   tflops(M, N, K, r.ours), tflops(M, N, K, r.cublas_fp32),
                   tflops(M, N, K, r.cublas_tf32), sp, r.beat ? 1 : 0, r.err,
                   bytes_moved(M, N, K));
      std::fflush(csv);
    }
  };

  if (!squares.empty()) {
    std::printf("\nSquare SGEMM (M=N=K)  — small/medium is the claim\n");
    for (auto sh : squares)
      run_shape(sh.M, sh.N, sh.K, "square", square_rows);
  }
  if (!skinny.empty()) {
    std::printf("\nSkinny SGEMM (N ≤ 128, large M=K)\n");
    for (auto sh : skinny)
      run_shape(sh.M, sh.N, sh.K, "skinny", skinny_rows);
  }

  if (!square_rows.empty())
    print_table("Square SGEMM", square_rows);
  if (!skinny_rows.empty())
    print_table("Skinny SGEMM  (N≤128)", skinny_rows);

  int wins = 0, total = 0;
  auto tally = [&](const std::vector<Row>& v) {
    for (const auto& r : v) {
      wins += r.beat;
      total++;
    }
  };
  tally(square_rows);
  tally(skinny_rows);
  std::printf("\nWins vs cuBLAS FP32: %d / %d\n", wins, total);

  if (csv)
    std::fclose(csv);
  CUBLAS_CHECK(cublasDestroy(handle));
  return 0;
}
