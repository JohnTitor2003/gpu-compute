cmake_minimum_required(VERSION 3.24)
project(gpu-compute LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

function(gpu_target name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR})
  target_link_libraries(${name} PRIVATE CUDA::cudart)
  target_compile_options(${name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo;--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/O2>>
  )
  set_target_properties(${name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
endfunction()

gpu_target(gemm_bench gemm/bench_gemm.cu)
target_link_libraries(gemm_bench PRIVATE CUDA::cublas)

gpu_target(pathtracer pathtracer/main.cu)
gpu_target(nbody nbody/main.cu)

install(TARGETS gemm_bench pathtracer nbody RUNTIME DESTINATION bin)
