#pragma once

#include "pathtracer/scene.cuh"

__device__ inline Vec3 sample_light(const SceneView& sc, const Hit& h, Rng& rng,
                                    Vec3 wo) {
  const Material mat = sc.mats[h.mat];
  if (mat.kind != MAT_DIFFUSE)
    return {0, 0, 0};

  const float u = rng.f();
  const float v = rng.f();
  Vec3 lp{sc.light_min.x + (sc.light_max.x - sc.light_min.x) * u,
          sc.light_min.y,
          sc.light_min.z + (sc.light_max.z - sc.light_min.z) * v};
  Vec3 to = lp - h.p;
  const float dist2 = length2(to);
  const float dist = sqrtf(dist2);
  Vec3 ldir = to / dist;
  const float n_dot_l = dot(h.n, ldir);
  const float ln_dot = dot(sc.light_n, -ldir);
  if (n_dot_l <= 0.f || ln_dot <= 0.f)
    return {0, 0, 0};
  if (occluded(sc, Ray{h.p, ldir}, dist - 1e-4f))
    return {0, 0, 0};
  // Lambert * NdotL * Le * (A * ln_dot / dist2) / 1  (pdf = 1/area)
  const float geom = n_dot_l * ln_dot * sc.light_area / dist2;
  return mat.albedo * (1.f / 3.14159265f) * sc.light_emit * geom;
}

__device__ inline bool scatter(const Material& mat, const Hit& h, Vec3 wo,
                               Rng& rng, Vec3& wi, Vec3& attn) {
  if (mat.kind == MAT_DIFFUSE) {
    wi = cosine_hemisphere(h.n, rng);
    attn = mat.albedo;  // cosine sampling cancels the pdf and NdotL
    return true;
  }
  if (mat.kind == MAT_METAL) {
    Vec3 r = reflect(wo, h.n);
    wi = normalize(r + cosine_hemisphere(h.n, rng) * mat.extra);
    if (dot(wi, h.n) <= 0.f)
      return false;
    attn = mat.albedo;
    return true;
  }
  if (mat.kind == MAT_GLASS) {
    const float ior = mat.extra > 0.f ? mat.extra : 1.5f;
    const float eta = h.front ? (1.f / ior) : ior;
    const float cos_i = fminf(-dot(wo, h.n), 1.f);
    const float reflect_p = schlick(cos_i, ior);
    Vec3 refr;
    if (rng.f() < reflect_p || !refract(wo, h.n, eta, refr)) {
      wi = reflect(wo, h.n);
    } else {
      wi = normalize(refr);
    }
    attn = mat.albedo;
    return true;
  }
  return false;
}

__device__ inline Vec3 trace_path(const SceneView& sc, Ray r, Rng rng,
                                  int max_bounce) {
  Vec3 radiance{0, 0, 0};
  Vec3 throughput{1, 1, 1};

  for (int bounce = 0; bounce < max_bounce; ++bounce) {
    Hit h;
    if (!scene_hit(sc, r, 1e-4f, 1e30f, h)) {
      radiance += throughput * Vec3{0.02f, 0.02f, 0.03f};
      break;
    }
    const Material mat = sc.mats[h.mat];
    if (mat.kind == MAT_LIGHT) {
      if (bounce == 0)
        radiance += throughput * mat.emission;
      break;
    }

    radiance += throughput * sample_light(sc, h, rng, r.d);

    Vec3 wi, attn;
    if (!scatter(mat, h, r.d, rng, wi, attn))
      break;
    throughput = throughput * attn;
    r = Ray{h.p, wi};

    if (bounce > 2) {
      const float p = fminf(0.95f, fmaxf(throughput.x, fmaxf(throughput.y, throughput.z)));
      if (rng.f() > p)
        break;
      throughput = throughput / p;
    }
  }
  return radiance;
}

__device__ inline Ray camera_ray(const Camera& cam, int x, int y, int w, int h,
                                 float u, float v, Rng& rng) {
  const float aspect = float(w) / float(h);
  Vec3 wdir = normalize(cam.origin - cam.look);
  Vec3 udir = normalize(cross(wdir, cam.up));
  Vec3 vdir = cross(udir, wdir);
  const float fl = 1.f / tanf(cam.fov * 0.5f);
  const float nx = ((x + u) / float(w) * 2.f - 1.f) * aspect;
  const float ny = (1.f - (y + v) / float(h) * 2.f);
  Vec3 aim = cam.origin + normalize(udir * nx + vdir * ny - wdir * fl) * cam.focus;
  Vec3 origin = cam.origin;
  if (cam.aperture > 0.f) {
    Vec3 d = disk_sample(rng) * (cam.aperture * 0.5f);
    origin = cam.origin + udir * d.x + vdir * d.y;
  }
  return Ray{origin, normalize(aim - origin)};
}

__global__ void path_kernel(SceneView sc, int w, int h, int spp, int seed,
                            int max_bounce, float* accum) {
  const int x = int(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = int(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= w || y >= h)
    return;
  const int pix = y * w + x;
  Rng rng{uint32_t(pix) * 1973u + uint32_t(seed) * 9277u + 1u};
  Vec3 sum{0, 0, 0};
  const int sx = spp > 16 ? 4 : (spp > 4 ? 2 : 1);
  const int sy = sx;
  const int strata = sx * sy;
  const int per = (spp + strata - 1) / strata;
  int taken = 0;
  for (int j = 0; j < sy && taken < spp; ++j) {
    for (int i = 0; i < sx && taken < spp; ++i) {
      for (int s = 0; s < per && taken < spp; ++s, ++taken) {
        const float u = (i + rng.f()) / float(sx);
        const float v = (j + rng.f()) / float(sy);
        Ray r = camera_ray(sc.cam, x, y, w, h, u, v, rng);
        sum += trace_path(sc, r, rng, max_bounce);
      }
    }
  }
  accum[pix * 3 + 0] += sum.x;
  accum[pix * 3 + 1] += sum.y;
  accum[pix * 3 + 2] += sum.z;
}

__host__ __device__ inline float reinhard(float x) { return x / (1.f + x); }

__global__ void tonemap_kernel(const float* accum, uint8_t* rgb, int n,
                               float inv_spp) {
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n)
    return;
  float r = accum[i * 3 + 0] * inv_spp;
  float g = accum[i * 3 + 1] * inv_spp;
  float b = accum[i * 3 + 2] * inv_spp;
  // ACES-ish fitted filmic, then gamma 2.2
  auto aces = [](float x) {
    const float a = 2.51f, bb = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    x = fmaxf(0.f, x);
    return fmaxf(0.f, (x * (a * x + bb)) / (x * (c * x + d) + e));
  };
  r = powf(fminf(1.f, aces(r)), 1.f / 2.2f);
  g = powf(fminf(1.f, aces(g)), 1.f / 2.2f);
  b = powf(fminf(1.f, aces(b)), 1.f / 2.2f);
  rgb[i * 3 + 0] = uint8_t(r * 255.f + 0.5f);
  rgb[i * 3 + 1] = uint8_t(g * 255.f + 0.5f);
  rgb[i * 3 + 2] = uint8_t(b * 255.f + 0.5f);
}
