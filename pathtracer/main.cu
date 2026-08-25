#include "common/cuda.hpp"
#include "common/png.hpp"
#include "pathtracer/trace.cuh"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  print_device();
  const int w = std::atoi(arg_value(argc, argv, "--width", "800").c_str());
  const int h = std::atoi(arg_value(argc, argv, "--height", "800").c_str());
  const int spp = std::atoi(arg_value(argc, argv, "--spp", "128").c_str());
  const int bounces = std::atoi(arg_value(argc, argv, "--bounces", "8").c_str());
  const std::string scene = arg_value(argc, argv, "--scene", "cornell");
  const std::string out = arg_value(argc, argv, "--out", "results/cornell.png");

  HostScene host;
  if (scene == "macro")
    host = make_cornell_macro();
  else if (scene == "atelier")
    host = make_atelier();
  else
    host = make_cornell();
  std::printf("Scene: %d spheres, %d triangles, %d BVH nodes, %d materials\n",
              int(host.spheres.size()), int(host.tris.size()),
              int(host.nodes.size()), int(host.mats.size()));

  Sphere* d_spheres = device_alloc<Sphere>(host.spheres.size());
  Triangle* d_tris = device_alloc<Triangle>(host.tris.size());
  BvhNode* d_nodes = device_alloc<BvhNode>(host.nodes.size());
  Material* d_mats = device_alloc<Material>(host.mats.size());
  CUDA_CHECK(cudaMemcpy(d_spheres, host.spheres.data(),
                        host.spheres.size() * sizeof(Sphere),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_tris, host.tris.data(),
                        host.tris.size() * sizeof(Triangle),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_nodes, host.nodes.data(),
                        host.nodes.size() * sizeof(BvhNode),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mats, host.mats.data(),
                        host.mats.size() * sizeof(Material),
                        cudaMemcpyHostToDevice));

  SceneView sc{};
  sc.spheres = d_spheres;
  sc.nspheres = int(host.spheres.size());
  sc.tris = d_tris;
  sc.ntris = int(host.tris.size());
  sc.nodes = d_nodes;
  sc.nnodes = int(host.nodes.size());
  sc.mats = d_mats;
  sc.light_min = host.light_min;
  sc.light_max = host.light_max;
  sc.light_n = host.light_n;
  sc.light_emit = host.light_emit;
  sc.light_area = host.light_area;
  sc.cam = host.cam;

  const int n = w * h;
  float* accum = device_alloc<float>(size_t(n) * 3);
  uint8_t* drgb = device_alloc<uint8_t>(size_t(n) * 3);
  CUDA_CHECK(cudaMemset(accum, 0, size_t(n) * 3 * sizeof(float)));

  ensure_parent_dir(out);
  dim3 block(16, 16);
  dim3 grid(div_up(w, 16), div_up(h, 16));

  GpuTimer t;
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto wall0 = std::chrono::steady_clock::now();
  t.record_start();
  path_kernel<<<grid, block>>>(sc, w, h, spp, 1, bounces, accum);
  t.record_stop();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto wall1 = std::chrono::steady_clock::now();
  const float gpu_ms = t.elapsed_ms();
  const float wall_ms =
      std::chrono::duration<float, std::milli>(wall1 - wall0).count();

  tonemap_kernel<<<(n + 255) / 256, 256>>>(accum, drgb, n, 1.f / float(spp));
  std::vector<uint8_t> rgb(size_t(n) * 3);
  CUDA_CHECK(cudaMemcpy(rgb.data(), drgb, rgb.size(), cudaMemcpyDeviceToHost));

  if (!png::write_rgb(out.c_str(), w, h, rgb.data())) {
    std::fprintf(stderr, "failed to write %s\n", out.c_str());
    return 1;
  }

  const double rays = double(n) * double(spp);
  const double mrays = rays / (gpu_ms * 1e3);
  std::printf("Wrote %s (%dx%d, %d spp, %d bounces)\n", out.c_str(), w, h, spp,
              bounces);
  std::printf("GPU  %.2f ms   wall %.2f ms   %.2f M camera-paths/s\n", gpu_ms,
              wall_ms, mrays);
  std::printf(
      "Primary rays: %.2f M   (bounce rays are extra; this is camera samples)\n",
      rays / 1e6);

  cudaFree(d_spheres);
  cudaFree(d_tris);
  cudaFree(d_nodes);
  cudaFree(d_mats);
  cudaFree(accum);
  cudaFree(drgb);
  return 0;
}
