# Potto applications
To run the microbenchmarks (Figures 15 and 16) run: 
`cd benchmarks`
`python plot_figures.py --load --all`

## Reproduce

To reproduce the figures in the paper, run the following commands:

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
