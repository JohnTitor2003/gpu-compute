#include "common/cuda.hpp"
#include "common/png.hpp"
#include "nbody/kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static float urand(unsigned& s) {
  s = s * 1664525u + 1013904223u;
  return float(s >> 8) * (1.f / 16777216.f);
}

// Two Plummer spheres on a collision course. Looks like a merger, sits in
// a unit-ish box so the splat projection is stable.
static void init_collision(std::vector<float4>& pos, std::vector<float4>& vel,
                           int n) {
  unsigned s = 42;
  const int n1 = n / 2;
  auto plummer = [&](int begin, int end, float cx, float cy, float cz, float vx) {
    const float mass = 1.f / float(n);
    for (int i = begin; i < end; ++i) {
      float r;
      do {
        const float x = std::pow(urand(s), 2.f / 3.f);
        r = 0.15f / std::sqrt(std::max(1e-6f, 1.f / x - 1.f));
      } while (r > 0.6f);
      const float theta = std::acos(2.f * urand(s) - 1.f);
      const float phi = 6.2831853f * urand(s);
      pos[i] = make_float4(cx + r * std::sin(theta) * std::cos(phi),
                           cy + r * std::sin(theta) * std::sin(phi),
                           cz + r * std::cos(theta), mass);
      vel[i] = make_float4(vx + (urand(s) - 0.5f) * 0.02f,
                           (urand(s) - 0.5f) * 0.02f,
                           (urand(s) - 0.5f) * 0.02f, 0.f);
    }
  };
  plummer(0, n1, -0.22f, 0.f, 0.f, 0.12f);
  plummer(n1, n, 0.22f, 0.04f, 0.f, -0.12f);
}

static void viridis(float t, uint8_t& r, uint8_t& g, uint8_t& b) {
  t = std::min(1.f, std::max(0.f, t));
  const float x = t;
  r = uint8_t(255.f * std::min(1.f, std::max(0.f, 2.739f * x - 0.164f * x * x -
                                                      1.668f + 0.1f)));
  g = uint8_t(255.f * std::min(1.f, std::max(0.f, -1.2f * (x - 0.75f) * (x - 0.75f) + 0.9f)));
  b = uint8_t(255.f * std::min(1.f, std::max(0.f, 1.4f - 2.2f * x + 0.3f)));
  // Better hand-tuned stops:
  const float stops[][3] = {
      {0.267f, 0.005f, 0.329f}, {0.283f, 0.141f, 0.458f},
      {0.254f, 0.265f, 0.530f}, {0.207f, 0.372f, 0.553f},
      {0.164f, 0.471f, 0.558f}, {0.128f, 0.567f, 0.551f},
      {0.135f, 0.659f, 0.518f}, {0.267f, 0.749f, 0.441f},
      {0.478f, 0.821f, 0.318f}, {0.741f, 0.873f, 0.150f}, {0.993f, 0.906f, 0.144f}};
  const float u = t * 10.f;
  const int i = std::min(9, int(u));
  const float f = u - float(i);
  r = uint8_t((stops[i][0] * (1 - f) + stops[i + 1][0] * f) * 255.f);
  g = uint8_t((stops[i][1] * (1 - f) + stops[i + 1][1] * f) * 255.f);
  b = uint8_t((stops[i][2] * (1 - f) + stops[i + 1][2] * f) * 255.f);
}

int main(int argc, char** argv) {
  print_device();
  const int n = std::atoi(arg_value(argc, argv, "--n", "16384").c_str());
  const int steps = std::atoi(arg_value(argc, argv, "--steps", "64").c_str());
  const float dt = std::atof(arg_value(argc, argv, "--dt", "0.002").c_str());
  const float eps = std::atof(arg_value(argc, argv, "--eps", "0.02").c_str());
  const std::string out = arg_value(argc, argv, "--out", "results/nbody.png");
  const int img = std::atoi(arg_value(argc, argv, "--image", "768").c_str());
  const bool skip_naive = arg_flag(argc, argv, "--skip-naive");

  if (n <= 0 || n % 128 != 0) {
    std::fprintf(stderr, "--n must be a positive multiple of 128 (got %d)\n", n);
    return 1;
  }

  std::vector<float4> hpos(n), hvel(n);
  init_collision(hpos, hvel, n);

  float4* pos = device_alloc<float4>(n);
  float4* vel = device_alloc<float4>(n);
  float4* acc = device_alloc<float4>(n);
  CUDA_CHECK(cudaMemcpy(pos, hpos.data(), n * sizeof(float4), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(vel, hvel.data(), n * sizeof(float4), cudaMemcpyHostToDevice));

  auto time_force = [&](auto launch, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i)
      launch(pos, acc, n, eps * eps, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer t;
    t.record_start();
    for (int i = 0; i < iters; ++i)
      launch(pos, acc, n, eps * eps, nullptr);
    t.record_stop();
    CUDA_CHECK(cudaGetLastError());
    return t.elapsed_ms() / float(iters);
  };

  const int warmup = 3;
  const int iters = n >= 32768 ? 5 : 10;
  const float tiled_ms = time_force(launch_tiled, warmup, iters);
  float naive_ms = 0.f;
  if (!skip_naive && n <= 32768)
    naive_ms = time_force(launch_naive, warmup, iters);

  // 20 FLOP/interaction is the usual accounting (3 sub, 3 mul-add for r2,
  // rsqrt, 2 mul for inv3, 3 mul for scale, 3 fma onto acc).
  const double pairs = double(n) * double(n);
  const double flop = pairs * 20.0;
  std::printf("\n## N-body force kernel  N=%d\n\n", n);
  std::printf("| kernel | ms | interactions/s | TFLOP/s |\n");
  std::printf("|--------|---:|---------------:|--------:|\n");
  std::printf("| tiled 128 | %.3f | %.2e | %.2f |\n", tiled_ms, pairs / (tiled_ms * 1e-3),
              flop / (tiled_ms * 1e9));
  if (naive_ms > 0.f) {
    std::printf("| naive     | %.3f | %.2e | %.2f |\n", naive_ms,
                pairs / (naive_ms * 1e-3), flop / (naive_ms * 1e9));
    std::printf("\nSpeedup vs naive: **%.2fx**\n", naive_ms / tiled_ms);
  }

  float *ke = device_alloc<float>(1), *pe = device_alloc<float>(1);
  auto energy = [&](float& out_ke, float& out_pe) {
    CUDA_CHECK(cudaMemset(ke, 0, 4));
    CUDA_CHECK(cudaMemset(pe, 0, 4));
    nbody_energy<<<(n + 255) / 256, 256>>>(pos, vel, n, eps * eps, ke, pe);
    CUDA_CHECK(cudaMemcpy(&out_ke, ke, 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&out_pe, pe, 4, cudaMemcpyDeviceToHost));
  };

  float ke0, pe0, ke1, pe1;
  energy(ke0, pe0);

  GpuTimer sim;
  sim.record_start();
  for (int s = 0; s < steps; ++s) {
    launch_tiled(pos, acc, n, eps * eps);
    nbody_integrate<<<(n + 255) / 256, 256>>>(pos, vel, acc, n, dt);
  }
  sim.record_stop();
  CUDA_CHECK(cudaGetLastError());
  const float sim_ms = sim.elapsed_ms();
  energy(ke1, pe1);

  std::printf("\nSimulated %d steps in %.2f ms  (%.2f steps/s)\n", steps, sim_ms,
              1000.f * steps / sim_ms);
  std::printf("Energy  E0=%.6f  E1=%.6f  rel drift=%.3e\n", ke0 + pe0, ke1 + pe1,
              std::abs((ke1 + pe1) - (ke0 + pe0)) / (std::abs(ke0 + pe0) + 1e-12));

  unsigned int* bins = device_alloc<unsigned int>(img * img);
  CUDA_CHECK(cudaMemset(bins, 0, img * img * sizeof(unsigned int)));
  nbody_splat<<<(n + 255) / 256, 256>>>(pos, n, img, img, 1.15f, bins);
  std::vector<unsigned int> hbins(img * img);
  CUDA_CHECK(cudaMemcpy(hbins.data(), bins, hbins.size() * 4,
                        cudaMemcpyDeviceToHost));
  unsigned int mx = 1;
  for (unsigned v : hbins)
    mx = std::max(mx, v);
  std::vector<uint8_t> rgb(size_t(img) * img * 3);
  for (int i = 0; i < img * img; ++i) {
    const float t = hbins[i] ? std::log1p(float(hbins[i])) / std::log1p(float(mx))
                             : 0.f;
    viridis(t, rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
  }
  ensure_parent_dir(out);
  if (!png::write_rgb(out.c_str(), img, img, rgb.data())) {
    std::fprintf(stderr, "failed to write %s\n", out.c_str());
    return 1;
  }
  std::printf("Wrote %s\n", out.c_str());

  cudaFree(pos);
  cudaFree(vel);
  cudaFree(acc);
  cudaFree(ke);
  cudaFree(pe);
  cudaFree(bins);
  return 0;
}
