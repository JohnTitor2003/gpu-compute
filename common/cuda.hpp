#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

inline int div_up(int a, int b) { return (a + b - 1) / b; }

inline void ensure_parent_dir(const std::string& path) {
  std::filesystem::path p(path);
  if (p.has_parent_path() && !p.parent_path().empty())
    std::filesystem::create_directories(p.parent_path());
}

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err__ = (expr);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s = %s\n", __FILE__, __LINE__,  \
                   #expr, cudaGetErrorString(err__));                          \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)



struct GpuTimer {
  cudaEvent_t start{};
  cudaEvent_t stop{};

  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;

  void record_start(cudaStream_t stream = nullptr) {
    CUDA_CHECK(cudaEventRecord(start, stream));
  }
  void record_stop(cudaStream_t stream = nullptr) {
    CUDA_CHECK(cudaEventRecord(stop, stream));
  }
  float elapsed_ms() {
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

inline void print_device() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  std::printf("GPU: %s  sm_%d%d  %.1f GB  %d SMs  bus %d-bit\n", p.name,
              p.major, p.minor,
              p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
              p.multiProcessorCount, p.memoryBusWidth);
}

template <typename T>
T* device_alloc(size_t n) {
  T* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  return p;
}

inline std::string arg_value(int argc, char** argv, const char* key,
                             const char* fallback) {
  const std::string prefix = std::string(key) + "=";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == key && i + 1 < argc)
      return argv[i + 1];
    if (a.rfind(prefix, 0) == 0)
      return a.substr(prefix.size());
  }
  return fallback;
}

inline bool arg_flag(int argc, char** argv, const char* key) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == key)
      return true;
  }
  return false;
}
