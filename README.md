# Potto applications
This repository contains the implementation for the applications in our differentiable programming language Potto. The applications include image stylization, fitting shader parameters, and optimizing a 3D shader. The implementation for the core to library is in https://github.com/divicomp/potto.

## Reproduce


###  Discontinuous shaders

To reproduce the results of discontinuous shaders (Figure 12), run:

```
cd graphics
python main.py --size large --mode primal
python main.py --size large --mode deriv
```

### Microbenchmarks

To reproduce the results of microbenchmarks between Potto and Teg (Figure 15 & 16), run:

```
cd benchmarks
python plot_figures.py --heaviside
python plot_figures.py --shader-swap
```
