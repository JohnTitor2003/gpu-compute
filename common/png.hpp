#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

// Minimal PNG writer: 8-bit RGB, uncompressed DEFLATE blocks. No libpng/zlib.
namespace png {

inline uint32_t crc32(const uint8_t* data, size_t n) {
  static uint32_t table[256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k)
        c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      table[i] = c;
    }
    init = true;
  }
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i)
    c = table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

inline uint32_t adler32(const uint8_t* data, size_t n) {
  uint32_t a = 1, b = 0;
  for (size_t i = 0; i < n; ++i) {
    a = (a + data[i]) % 65521u;
    b = (b + a) % 65521u;
  }
  return (b << 16) | a;
}

inline void put_be32(std::vector<uint8_t>& o, uint32_t v) {
  o.push_back(uint8_t(v >> 24));
  o.push_back(uint8_t(v >> 16));
  o.push_back(uint8_t(v >> 8));
  o.push_back(uint8_t(v));
}

inline void chunk(std::vector<uint8_t>& o, const char type[4],
                  const uint8_t* data, size_t n) {
  put_be32(o, uint32_t(n));
  size_t type_off = o.size();
  o.insert(o.end(), type, type + 4);
  o.insert(o.end(), data, data + n);
  uint32_t c = crc32(o.data() + type_off, 4 + n);
  put_be32(o, c);
}

inline bool write_rgb(const char* path, int w, int h, const uint8_t* rgb) {
  std::vector<uint8_t> raw(size_t(h) * (1 + size_t(w) * 3));
  for (int y = 0; y < h; ++y) {
    raw[size_t(y) * (1 + size_t(w) * 3)] = 0;
    for (int x = 0; x < w; ++x) {
      const size_t s = (size_t(y) * w + x) * 3;
      const size_t d = size_t(y) * (1 + size_t(w) * 3) + 1 + size_t(x) * 3;
      raw[d] = rgb[s];
      raw[d + 1] = rgb[s + 1];
      raw[d + 2] = rgb[s + 2];
    }
  }

  std::vector<uint8_t> z;
  z.push_back(0x78);
  z.push_back(0x01);
  size_t off = 0;
  while (off < raw.size()) {
    const size_t take = std::min<size_t>(65535, raw.size() - off);
    const bool last = (off + take == raw.size());
    z.push_back(last ? 1 : 0);
    z.push_back(uint8_t(take));
    z.push_back(uint8_t(take >> 8));
    const uint16_t nlen = uint16_t(~uint16_t(take));
    z.push_back(uint8_t(nlen));
    z.push_back(uint8_t(nlen >> 8));
    z.insert(z.end(), raw.begin() + off, raw.begin() + off + take);
    off += take;
  }
  put_be32(z, adler32(raw.data(), raw.size()));

  std::vector<uint8_t> png;
  const uint8_t sig[] = {137, 80, 78, 71, 13, 10, 26, 10};
  png.insert(png.end(), sig, sig + 8);

  uint8_t ihdr[13] = {};
  ihdr[0] = uint8_t(w >> 24);
  ihdr[1] = uint8_t(w >> 16);
  ihdr[2] = uint8_t(w >> 8);
  ihdr[3] = uint8_t(w);
  ihdr[4] = uint8_t(h >> 24);
  ihdr[5] = uint8_t(h >> 16);
  ihdr[6] = uint8_t(h >> 8);
  ihdr[7] = uint8_t(h);
  ihdr[8] = 8;
  ihdr[9] = 2;
  chunk(png, "IHDR", ihdr, 13);
  chunk(png, "IDAT", z.data(), z.size());
  chunk(png, "IEND", nullptr, 0);

  FILE* f = nullptr;
#if defined(_MSC_VER)
  fopen_s(&f, path, "wb");
#else
  f = std::fopen(path, "wb");
#endif
  if (!f)
    return false;
  const size_t n = std::fwrite(png.data(), 1, png.size(), f);
  std::fclose(f);
  return n == png.size();
}

}  // namespace png
