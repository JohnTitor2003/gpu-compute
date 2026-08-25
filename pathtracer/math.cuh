#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

struct Vec3 {
  float x, y, z;
  __host__ __device__ Vec3() : x(0), y(0), z(0) {}
  __host__ __device__ Vec3(float v) : x(v), y(v), z(v) {}
  __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
};

__host__ __device__ inline Vec3 operator+(Vec3 a, Vec3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
__host__ __device__ inline Vec3 operator-(Vec3 a, Vec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
__host__ __device__ inline Vec3 operator-(Vec3 a) { return {-a.x, -a.y, -a.z}; }
__host__ __device__ inline Vec3 operator*(Vec3 a, Vec3 b) {
  return {a.x * b.x, a.y * b.y, a.z * b.z};
}
__host__ __device__ inline Vec3 operator*(Vec3 a, float s) {
  return {a.x * s, a.y * s, a.z * s};
}
__host__ __device__ inline Vec3 operator*(float s, Vec3 a) { return a * s; }
__host__ __device__ inline Vec3 operator/(Vec3 a, float s) {
  return a * (1.f / s);
}
__host__ __device__ inline Vec3& operator+=(Vec3& a, Vec3 b) {
  a = a + b;
  return a;
}

__host__ __device__ inline float dot(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__host__ __device__ inline float length2(Vec3 a) { return dot(a, a); }
__host__ __device__ inline float length(Vec3 a) { return sqrtf(length2(a)); }
__host__ __device__ inline Vec3 normalize(Vec3 a) {
  return a * rsqrtf(length2(a) + 1e-20f);
}
__host__ __device__ inline Vec3 min3(Vec3 a, Vec3 b) {
  return {fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)};
}
__host__ __device__ inline Vec3 max3(Vec3 a, Vec3 b) {
  return {fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)};
}
__host__ __device__ inline Vec3 reflect(Vec3 v, Vec3 n) {
  return v - n * (2.f * dot(v, n));
}

struct Ray {
  Vec3 o, d;
  __host__ __device__ Vec3 at(float t) const { return o + d * t; }
};

// Tiny PCG-ish hash. One state per thread; enough for stratified pixel samples.
struct Rng {
  uint32_t s;
  __host__ __device__ explicit Rng(uint32_t seed) : s(seed) {}
  __host__ __device__ uint32_t u32() {
    s = s * 747796405u + 2891336453u;
    uint32_t x = s;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
  }
  __host__ __device__ float f() { return (u32() >> 8) * (1.f / 16777216.f); }
};

__device__ inline Vec3 disk_sample(Rng& rng) {
  const float r = sqrtf(rng.f());
  const float a = 6.2831853f * rng.f();
  return {r * cosf(a), r * sinf(a), 0.f};
}

__device__ inline Vec3 cosine_hemisphere(Vec3 n, Rng& rng) {
  const float r1 = rng.f();
  const float r2 = rng.f();
  const float phi = 6.2831853f * r1;
  const float r = sqrtf(r2);
  const float x = r * cosf(phi);
  const float y = r * sinf(phi);
  const float z = sqrtf(fmaxf(0.f, 1.f - r2));
  Vec3 t = fabsf(n.y) < 0.999f ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
  Vec3 b = normalize(cross(n, t));
  t = cross(b, n);
  return normalize(t * x + b * y + n * z);
}

__host__ __device__ inline bool refract(Vec3 v, Vec3 n, float eta, Vec3& out) {
  const float cos_i = fminf(-dot(v, n), 1.f);
  const float sin2_t = eta * eta * (1.f - cos_i * cos_i);
  if (sin2_t > 1.f)
    return false;
  const float cos_t = sqrtf(1.f - sin2_t);
  out = v * eta + n * (eta * cos_i - cos_t);
  return true;
}

__host__ __device__ inline float schlick(float cos_i, float ior) {
  float r0 = (1.f - ior) / (1.f + ior);
  r0 *= r0;
  const float m = 1.f - cos_i;
  return r0 + (1.f - r0) * m * m * m * m * m;
}
