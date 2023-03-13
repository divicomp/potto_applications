# Benchmarks

This folder contains benchmarks of our differentiable programming language Potto. The graphics applications can be found at [graphics](../grahpics). The implementation for Potto can be found at https://github.com/divicomp/potto.


## Microbenchmarks

We design two microbenchmarks to compare Potto with [Teg](https://github.com/ChezJrk/Teg).
We compare compile time (time to calculate the derivative), runtime, total time (both compilation and runtime), and code size.

### Increasing the number of parametric discontinuities

We study how the performance of Potto and Teg scale as we increase the number of differentiated parametric discontinuities.
To do this, we increase the numbers of conditionals added together. This results in an increasing numbers of Dirac deltas in derivative.

See `run_potto_heaviside_microbenchmark` in `potto_microbenchmarks.py` and `run_teg_heaviside_microbenchmark` in `teg_microbenchmarks.py` for Potto and Teg implementations of the benchmark, respectively.


### Separate compilation

To study the impact of separate compilation. We design programs that generate expressions with 90 parametric discontinuities and scale the number of times we swap the shaders.

See `run_potto_shader_swap_microbenchmark` in `potto_microbenchmarks.py` and `run_teg_shader_swap_microbenchmark` in `teg_microbenchmarks.py` for Potto and Teg implementations of the benchmark, respectively.


### Reproduce

To reproduce figures of these two microbenchmarks (Fig.9 and Fig.10 in the paper), run:


```
python plot_figures.py
```

Notice that for the separate compilation benchmark, the compilation time of Teg reaches timeout very quickly so we manually set its maximum number of shader swaps to 15.


## Image stylization
We compare Potto and Teg using the image stylization example from Teg paper in FIgures 5 and 6. Similar to microbenchmarks, we report compilation time, evaluation time, total time, and code size for linear and quadratic shader stylization.

## Reproduce

To reproduce Table 1, run:

```
python potto_rasterization.py
python teg_rasterization.py
```
