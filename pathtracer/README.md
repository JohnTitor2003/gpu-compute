# GPU path tracer

CUDA path tracer: mixed analytic / mesh primitives, CPU-built BVH, next-event estimation, thin-lens depth of field.

![Cornell](../results/cornell.png)
![Atelier](../results/atelier.png)

## Scenes

| `--scene` | |
|---|---|
| `cornell` | unit Cornell box, metal + glass spheres, 320-triangle icosphere |
| `macro` | same geometry, camera inside the box, aperture on the dielectric |
| `atelier` | checker floor, gold / chrome / glass / ruby, icosphere, thin-lens |

## Algorithm

- One thread per pixel, stratified samples
- Thin lens: uniform disk on the aperture, focus plane at `cam.focus`
- Up to 8 bounces, Russian roulette after bounce 2
- Diffuse: cosine hemisphere + NEE (emission only on bounce 0)
- Metal: reflect + cosine fuzz
- Glass: Schlick between reflect and refract
- BVH: median split, 4-triangle leaves, 32-wide stack on device
- Tone map: fitted ACES + γ 2.2

```powershell
./build/bin/pathtracer.exe --scene cornell --width 800 --height 800 --spp 256 --out results/cornell.png
./build/bin/pathtracer.exe --scene atelier --width 960 --height 640 --spp 384 --out results/atelier.png
```
