# Potto Graphics

This folder contains code of shaders implemented using Potto. The benchmark code can be found at [benchmarks](../benchmarks). The implementation for Potto can be found at https://github.com/divicomp/potto.


## Truncated Lambert Shader

The truncated Lambert shader has a discontinuity inside the triangle, based on the amount of received light from the light source. If the light intensity is below certain threshold, the triangle is colored red, otherwise it is colored following Lambertian reflectance model.

### Reproduce
To reproduce the primal and derivative figures of this shader, run following commands.
```
python main.py --size large --mode primal
python main.py --size large --mode deriv
```
Note that rendering a full large size image (40x40) could be very time consuming. We also provide preview options like `--size small` and `--size medium` to render a 4x4 or 16x16 preview image.
