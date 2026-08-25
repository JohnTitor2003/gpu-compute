#pragma once

#include "pathtracer/math.cuh"

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

enum MatKind : int { MAT_DIFFUSE = 0, MAT_METAL = 1, MAT_GLASS = 2, MAT_LIGHT = 3 };

struct Material {
  int kind;
  Vec3 albedo;
  Vec3 emission;
  float extra;  // metal fuzz, glass IOR
};

struct Sphere {
  Vec3 c;
  float r;
  int mat;
};

struct Triangle {
  Vec3 a, b, c;
  int mat;
};

struct Aabb {
  Vec3 bmin, bmax;
  __host__ __device__ Aabb()
      : bmin{1e30f, 1e30f, 1e30f}, bmax{-1e30f, -1e30f, -1e30f} {}
  __host__ __device__ void grow(Vec3 p) {
    bmin = min3(bmin, p);
    bmax = max3(bmax, p);
  }
  __host__ __device__ void grow(const Aabb& o) {
    grow(o.bmin);
    grow(o.bmax);
  }
};

struct BvhNode {
  Vec3 bmin, bmax;
  int left;   // child index, or first triangle if leaf
  int right;  // right child, or triangle count if leaf
  int leaf;   // 1 = leaf
};

struct Hit {
  float t;
  Vec3 p;
  Vec3 n;
  int mat;
  bool front;
};

__host__ __device__ inline bool aabb_hit(Vec3 bmin, Vec3 bmax, Ray r, float tmin,
                                         float tmax) {
  for (int i = 0; i < 3; ++i) {
    const float o = i == 0 ? r.o.x : i == 1 ? r.o.y : r.o.z;
    const float d = i == 0 ? r.d.x : i == 1 ? r.d.y : r.d.z;
    const float mn = i == 0 ? bmin.x : i == 1 ? bmin.y : bmin.z;
    const float mx = i == 0 ? bmax.x : i == 1 ? bmax.y : bmax.z;
    const float inv = 1.f / d;
    float t0 = (mn - o) * inv;
    float t1 = (mx - o) * inv;
    if (inv < 0.f) {
      float tmp = t0;
      t0 = t1;
      t1 = tmp;
    }
    tmin = t0 > tmin ? t0 : tmin;
    tmax = t1 < tmax ? t1 : tmax;
    if (tmax <= tmin)
      return false;
  }
  return true;
}

__host__ __device__ inline bool tri_hit(const Triangle& tr, Ray r, float tmin,
                                        float tmax, Hit& h) {
  const Vec3 e1 = tr.b - tr.a;
  const Vec3 e2 = tr.c - tr.a;
  const Vec3 p = cross(r.d, e2);
  const float det = dot(e1, p);
  if (fabsf(det) < 1e-8f)
    return false;
  const float inv = 1.f / det;
  const Vec3 s = r.o - tr.a;
  const float u = dot(s, p) * inv;
  if (u < 0.f || u > 1.f)
    return false;
  const Vec3 q = cross(s, e1);
  const float v = dot(r.d, q) * inv;
  if (v < 0.f || u + v > 1.f)
    return false;
  const float t = dot(e2, q) * inv;
  if (t <= tmin || t >= tmax)
    return false;
  h.t = t;
  h.p = r.at(t);
  Vec3 n = cross(e1, e2);
  h.front = dot(n, r.d) < 0.f;
  h.n = normalize(h.front ? n : -n);
  h.mat = tr.mat;
  return true;
}

__host__ __device__ inline bool sphere_hit(const Sphere& sp, Ray r, float tmin,
                                           float tmax, Hit& h) {
  const Vec3 oc = r.o - sp.c;
  const float a = length2(r.d);
  const float b = dot(oc, r.d);
  const float c = length2(oc) - sp.r * sp.r;
  const float disc = b * b - a * c;
  if (disc < 0.f)
    return false;
  const float s = sqrtf(disc);
  float t = (-b - s) / a;
  if (t <= tmin || t >= tmax) {
    t = (-b + s) / a;
    if (t <= tmin || t >= tmax)
      return false;
  }
  h.t = t;
  h.p = r.at(t);
  h.n = (h.p - sp.c) / sp.r;
  h.front = dot(r.d, h.n) < 0.f;
  if (!h.front)
    h.n = -h.n;
  h.mat = sp.mat;
  return true;
}

struct Camera {
  Vec3 origin{0.5f, 0.52f, -1.15f};
  Vec3 look{0.5f, 0.45f, 0.5f};
  Vec3 up{0, 1, 0};
  float fov = 0.62f;
  float aperture = 0.f;
  float focus = 1.6f;
};

struct SceneView {
  const Sphere* spheres;
  int nspheres;
  const Triangle* tris;
  int ntris;
  const BvhNode* nodes;
  int nnodes;
  const Material* mats;
  Vec3 light_min, light_max;
  Vec3 light_n;
  Vec3 light_emit;
  float light_area;
  Camera cam;
};

__device__ inline bool bvh_hit(const SceneView& sc, Ray r, float tmin, float tmax,
                               Hit& out) {
  int stack[32];
  int sp = 0;
  stack[sp++] = 0;
  bool hit = false;
  Hit tmp;
  while (sp) {
    const BvhNode n = sc.nodes[stack[--sp]];
    if (!aabb_hit(n.bmin, n.bmax, r, tmin, tmax))
      continue;
    if (n.leaf) {
      for (int i = 0; i < n.right; ++i) {
        if (tri_hit(sc.tris[n.left + i], r, tmin, tmax, tmp)) {
          tmax = tmp.t;
          out = tmp;
          hit = true;
        }
      }
    } else {
      if (sp < 30) {
        stack[sp++] = n.left;
        stack[sp++] = n.right;
      }
    }
  }
  return hit;
}

__device__ inline bool scene_hit(const SceneView& sc, Ray r, float tmin,
                                 float tmax, Hit& out) {
  bool hit = false;
  Hit tmp;
  for (int i = 0; i < sc.nspheres; ++i) {
    if (sphere_hit(sc.spheres[i], r, tmin, tmax, tmp)) {
      tmax = tmp.t;
      out = tmp;
      hit = true;
    }
  }
  if (sc.nnodes > 0 && bvh_hit(sc, r, tmin, tmax, tmp)) {
    out = tmp;
    hit = true;
  }
  return hit;
}

__device__ inline bool occluded(const SceneView& sc, Ray r, float tmax) {
  Hit h;
  if (!scene_hit(sc, r, 1e-4f, tmax, h))
    return false;
  return sc.mats[h.mat].kind != MAT_LIGHT;
}

struct HostScene {
  std::vector<Sphere> spheres;
  std::vector<Triangle> tris;
  std::vector<BvhNode> nodes;
  std::vector<Material> mats;
  Vec3 light_min, light_max, light_n, light_emit;
  float light_area = 0.f;
  Camera cam;

  int add_mat(int kind, Vec3 albedo, Vec3 emit = {0, 0, 0}, float extra = 0.f) {
    mats.push_back({kind, albedo, emit, extra});
    return int(mats.size()) - 1;
  }

  void add_quad(Vec3 a, Vec3 b, Vec3 c, Vec3 d, int mat) {
    tris.push_back({a, b, c, mat});
    tris.push_back({a, c, d, mat});
  }

  void add_icosphere(Vec3 center, float radius, int mat, int subdiv) {
    const float t = (1.f + std::sqrt(5.f)) * 0.5f;
    std::vector<Vec3> v = {
        {-1, t, 0},  {1, t, 0},  {-1, -t, 0}, {1, -t, 0},
        {0, -1, t},  {0, 1, t},  {0, -1, -t}, {0, 1, -t},
        {t, 0, -1},  {t, 0, 1},  {-t, 0, -1}, {-t, 0, 1},
    };
    for (auto& p : v)
      p = normalize(p);
    std::vector<int> idx = {0, 11, 5,  0, 5,  1,  0, 1,  7,  0, 7,  10, 0, 10, 11,
                            1, 5,  9,  5, 11, 4,  11, 10, 2,  10, 7, 6,  7, 1,  8,
                            3, 9,  4,  3, 4,  2,  3,  2,  6,  3,  6, 8,  3, 8,  9,
                            4, 9,  5,  2, 4,  11, 6,  2,  10, 8,  6, 7,  9, 8,  1};
    auto midpoint = [&](int a, int b) {
      Vec3 m = normalize(v[a] + v[b]);
      v.push_back(m);
      return int(v.size()) - 1;
    };
    for (int s = 0; s < subdiv; ++s) {
      std::vector<int> next;
      next.reserve(idx.size() * 4);
      for (size_t i = 0; i < idx.size(); i += 3) {
        int a = idx[i], b = idx[i + 1], c = idx[i + 2];
        int ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
        next.insert(next.end(), {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca});
      }
      idx.swap(next);
    }
    for (size_t i = 0; i < idx.size(); i += 3) {
      tris.push_back({v[idx[i]] * radius + center,
                      v[idx[i + 1]] * radius + center,
                      v[idx[i + 2]] * radius + center, mat});
    }
  }

  void build_bvh() {
    nodes.clear();
    if (tris.empty())
      return;
    std::vector<int> order(tris.size());
    for (int i = 0; i < int(order.size()); ++i)
      order[i] = i;
    std::vector<Triangle> sorted(tris.size());
    nodes.reserve(tris.size() * 2);

    auto centroid = [](const Triangle& t) {
      return (t.a + t.b + t.c) * (1.f / 3.f);
    };
    auto bounds_of = [](const Triangle& t) {
      Aabb b;
      b.grow(t.a);
      b.grow(t.b);
      b.grow(t.c);
      return b;
    };

    auto rec = [&](auto&& rec, int begin, int end) -> int {
      BvhNode node{};
      Aabb box;
      for (int i = begin; i < end; ++i)
        box.grow(bounds_of(tris[order[i]]));
      node.bmin = box.bmin;
      node.bmax = box.bmax;
      const int count = end - begin;
      const int id = int(nodes.size());
      nodes.push_back(node);
      if (count <= 4) {
        nodes[id].leaf = 1;
        nodes[id].left = begin;
        nodes[id].right = count;
        return id;
      }
      int axis = 0;
      Vec3 ext = box.bmax - box.bmin;
      if (ext.y > ext.x)
        axis = 1;
      if (ext.z > (axis == 0 ? ext.x : ext.y))
        axis = 2;
      std::sort(order.begin() + begin, order.begin() + end,
                [&](int i, int j) {
                  Vec3 ci = centroid(tris[i]);
                  Vec3 cj = centroid(tris[j]);
                  float a = axis == 0 ? ci.x : axis == 1 ? ci.y : ci.z;
                  float b = axis == 0 ? cj.x : axis == 1 ? cj.y : cj.z;
                  return a < b;
                });
      const int mid = (begin + end) / 2;
      const int L = rec(rec, begin, mid);
      const int R = rec(rec, mid, end);
      nodes[id].leaf = 0;
      nodes[id].left = L;
      nodes[id].right = R;
      return id;
    };

    rec(rec, 0, int(tris.size()));
    for (int i = 0; i < int(order.size()); ++i)
      sorted[i] = tris[order[i]];
    // Rebuild leaves so left is an index into the *sorted* array. The recursive
    // build stored `begin` into the original `order` permutation; after we
    // compact triangles into `sorted[i] = tris[order[i]]`, leaf.left must be
    // the compact range [begin, begin+count), which it already is.
    tris.swap(sorted);
  }
};

inline HostScene make_cornell() {
  HostScene s;
  const int white = s.add_mat(MAT_DIFFUSE, {0.73f, 0.73f, 0.73f});
  const int red = s.add_mat(MAT_DIFFUSE, {0.65f, 0.05f, 0.05f});
  const int green = s.add_mat(MAT_DIFFUSE, {0.12f, 0.45f, 0.15f});
  const int light = s.add_mat(MAT_LIGHT, {0, 0, 0}, {22.f, 21.f, 19.f});
  const int metal = s.add_mat(MAT_METAL, {0.95f, 0.93f, 0.88f}, {0, 0, 0}, 0.05f);
  const int glass = s.add_mat(MAT_GLASS, {1, 1, 1}, {0, 0, 0}, 1.5f);
  const int blue = s.add_mat(MAT_DIFFUSE, {0.25f, 0.35f, 0.75f});

  // Cornell box in [0,1]^3, camera looks toward +z from z<0? We'll use
  // physically common layout: y up, camera at z= -something looking +z,
  // room [0,555] scaled to ~1. Use 1-unit box.
  const float e = 1.f;
  Vec3 A{0, 0, 0}, B{e, 0, 0}, C{e, 0, e}, D{0, 0, e};
  Vec3 E{0, e, 0}, F{e, e, 0}, G{e, e, e}, H{0, e, e};
  s.add_quad(A, B, C, D, white);  // floor
  s.add_quad(E, H, G, F, white);  // ceiling
  s.add_quad(D, C, G, H, white);  // back
  s.add_quad(A, D, H, E, red);    // left
  s.add_quad(B, F, G, C, green);  // right

  // Area light on the ceiling.
  const float x0 = 0.35f, x1 = 0.65f, z0 = 0.35f, z1 = 0.65f, y = 0.999f;
  s.add_quad({x0, y, z0}, {x1, y, z0}, {x1, y, z1}, {x0, y, z1}, light);
  s.light_min = {x0, y, z0};
  s.light_max = {x1, y, z1};
  s.light_n = {0, -1, 0};
  s.light_emit = {22.f, 21.f, 19.f};
  s.light_area = (x1 - x0) * (z1 - z0);

  s.spheres.push_back({{0.27f, 0.22f, 0.35f}, 0.22f, metal});
  s.spheres.push_back({{0.72f, 0.20f, 0.65f}, 0.20f, glass});
  s.add_icosphere({0.50f, 0.12f, 0.22f}, 0.12f, blue, 2);

  s.build_bvh();
  return s;
}

inline HostScene make_cornell_macro() {
  HostScene s = make_cornell();
  s.cam.origin = {0.38f, 0.28f, -0.05f};
  s.cam.look = {0.72f, 0.20f, 0.62f};
  s.cam.fov = 0.50f;
  s.cam.aperture = 0.024f;
  s.cam.focus = length(s.cam.look - s.cam.origin);
  return s;
}

inline HostScene make_atelier() {
  HostScene s;
  const int light = s.add_mat(MAT_LIGHT, {0, 0, 0}, {40.f, 38.f, 34.f});
  const int ivory = s.add_mat(MAT_DIFFUSE, {0.82f, 0.78f, 0.70f});
  const int charcoal = s.add_mat(MAT_DIFFUSE, {0.07f, 0.07f, 0.075f});
  const int gold = s.add_mat(MAT_METAL, {1.00f, 0.78f, 0.34f}, {0, 0, 0}, 0.08f);
  const int chrome = s.add_mat(MAT_METAL, {0.95f, 0.95f, 0.97f}, {0, 0, 0}, 0.02f);
  const int glass = s.add_mat(MAT_GLASS, {1, 1, 1}, {0, 0, 0}, 1.5f);
  const int ruby = s.add_mat(MAT_GLASS, {0.85f, 0.15f, 0.18f}, {0, 0, 0}, 1.5f);
  const int clay = s.add_mat(MAT_DIFFUSE, {0.55f, 0.22f, 0.16f});

  const float half = 2.4f;
  const int n = 12;
  const float tile = (2.f * half) / n;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const float x0 = -half + i * tile, z0 = -half + j * tile;
      const int mat = ((i + j) & 1) ? ivory : charcoal;
      s.add_quad({x0, 0, z0}, {x0 + tile, 0, z0}, {x0 + tile, 0, z0 + tile},
                 {x0, 0, z0 + tile}, mat);
    }
  }
  s.add_quad({-half, 0, -half}, {half, 0, -half}, {half, 2.2f, -half},
             {-half, 2.2f, -half}, charcoal);
  const float lx0 = -0.35f, lx1 = 0.35f, lz0 = -0.15f, lz1 = 0.45f, ly = 2.15f;
  s.add_quad({lx0, ly, lz0}, {lx1, ly, lz0}, {lx1, ly, lz1}, {lx0, ly, lz1}, light);
  s.light_min = {lx0, ly, lz0};
  s.light_max = {lx1, ly, lz1};
  s.light_n = {0, -1, 0};
  s.light_emit = {40.f, 38.f, 34.f};
  s.light_area = (lx1 - lx0) * (lz1 - lz0);

  s.spheres.push_back({{0.00f, 0.38f, 0.20f}, 0.38f, glass});
  s.spheres.push_back({{-0.85f, 0.28f, 0.55f}, 0.28f, gold});
  s.spheres.push_back({{0.95f, 0.22f, 0.70f}, 0.22f, chrome});
  s.spheres.push_back({{0.45f, 0.14f, -0.15f}, 0.14f, ruby});
  s.spheres.push_back({{-0.25f, 0.12f, 0.95f}, 0.12f, clay});
  s.add_icosphere({-1.15f, 0.18f, -0.20f}, 0.18f, clay, 2);

  s.cam.origin = {0.15f, 0.72f, 2.35f};
  s.cam.look = {0.05f, 0.28f, 0.25f};
  s.cam.fov = 0.55f;
  s.cam.aperture = 0.045f;
  s.cam.focus = length(s.cam.look - s.cam.origin);
  s.build_bvh();
  return s;
}
