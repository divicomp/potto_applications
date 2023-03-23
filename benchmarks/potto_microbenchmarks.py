import numpy as np
import sys
import matplotlib.pyplot as plt

from potto import Shift
from potto import (
    Var,
    TegVar,
    Heaviside,
    Const,
    Int,
    Sym,
    Function,
    App,
)
from potto import deriv
from potto import evaluate, get_ast_size
from potto import VarVal
from potto import BoundedLebesgue

import time


sys.setrecursionlimit(10**6)

NUM_SHADER_RUNS = 3
NUM_HEAVISIDE_RUNS = 10

def run_potto_shader_swap_microbenchmark(num_shader_swap=10, num_samples=10):
    x = TegVar("x")
    measure = BoundedLebesgue(-1, 1, x)
    a = Var("a")

    # Construct the big expression
    # \sum_i=10^100 H(x + ai)
    # where a_i = 1/i and da_i = 0
    eval_times, compile_times, ast_sizes = [], [], []
    num_shader_swaps = list(range(0, num_shader_swap, 1))
    for n in num_shader_swaps:
        big = Const(0)
        ctx = {}
        var_val = {}
        f = Var('f')
        for i in range(10, 100):
            new_a = Var(f"a{i}")
            var_val[new_a.name] = 1 / i
            big += Heaviside(Shift((new_a,), (x,), Const(1))) * App(
                f,
                (
                    new_a,
                    x,
                ),
            )
            dai = Sym(f"da{i}")
            ctx[new_a.name] = dai
            var_val[dai] = 1

        x0 = TegVar('x0')
        a0 = Var('a0')
        dx0_name = Sym('x0')
        da0_name = Sym('a0')
        ctx[x0.name] = dx0_name
        ctx[a0.name] = da0_name
        shader1 = Function(
            (
                a0,
                x0,
            ),
            a0 + x0,
        )
        shader2 = Function(
            (
                a0,
                x0,
            ),
            a0 - x0,
        )

        var_val = VarVal(var_val)
        integral = Function((f,), Int(big, measure))

        compile_times.append([])
        eval_times.append([])
        for _ in range(NUM_SHADER_RUNS):
            # with separate compilation
            start = time.time()
            di = deriv(integral, ctx)
            dexpr1s = []
            for _ in range(n):
                ds1 = deriv(shader1, ctx)
                dexpr1s.append(App(di, (shader1, ds1)))
            ds2 = deriv(shader2, ctx)
            dexpr2 = App(di, (shader2, ds2))
            dexpr = dexpr2
            for dexpr1 in dexpr1s:
                dexpr += dexpr1
            end = time.time()
            compile_time = end - start
            compile_times[n].append(compile_time)

            start = time.time()
            _ = evaluate(dexpr, var_val, num_samples)
            end = time.time()
            eval_time = end - start
            eval_times[n].append(eval_time)
        assert(len(compile_times[n]) == NUM_RUNS)
        assert(len(eval_times[n]) == NUM_RUNS)
        size = get_ast_size(dexpr)
        print(f"AST size: {size}")
        ast_sizes.append(size)

    print(num_shader_swaps)
    print(compile_times)
    print(eval_times)
    print(ast_sizes)
    return num_shader_swaps, compile_times, eval_times, ast_sizes


def run_potto_heaviside_microbenchmark(num_heaviside=10, num_samples=10):
    x = TegVar("x")
    measure = BoundedLebesgue(-1, 1, x)
    a = Var("a")

    # Construct the big expression
    # \sum_i=10^100 H(x + ai)
    # where a_i = 1/i and da_i = 0
    eval_times, compile_times, ast_sizes = [], [], []
    num_heavisides = list(range(0, num_heaviside, 1))
    for n in num_heavisides:
        big = Const(0)
        ctx = {}
        var_val = {}
        f = Var('f')
        for i in range(10, 10 + n):
            new_a = Var(f"a{i}")
            var_val[new_a.name] = 1 / i
            big += Heaviside(Shift((new_a,), (x,), Const(1))) * App(
                f,
                (
                    new_a,
                    x,
                ),
            )
            dai = Sym(f"da{i}")
            ctx[new_a.name] = dai
            var_val[dai] = 1

        x0 = TegVar('x0')
        a0 = Var('a0')
        dx0_name = Sym('x0')
        da0_name = Sym('a0')
        ctx[x0.name] = dx0_name
        ctx[a0.name] = da0_name
        shader1 = Function(
            (
                a0,
                x0,
            ),
            a0 + x0,
        )
        shader2 = Function(
            (
                a0,
                x0,
            ),
            a0 - x0,
        )

        var_val = VarVal(var_val)
        integral = Function((f,), Int(big, measure))

        compile_times.append([])
        eval_times.append([])
        for _ in range(NUM_HEAVISIDE_RUNS):
            # with separate compilation
            start = time.time()
            di = deriv(integral, ctx)
            ds1 = deriv(shader1, ctx)
            dexpr1 = App(di, (shader1, ds1))
            ds2 = deriv(shader2, ctx)
            dexpr2 = App(di, (shader2, ds2))
            dexpr = dexpr2
            dexpr += dexpr1
            end = time.time()
            compile_time = end - start
            compile_times[n].append(compile_time)

            start = time.time()
            _ = evaluate(dexpr, var_val, num_samples)
            end = time.time()
            eval_time = (end - start)
            eval_times[n].append(eval_time)
        assert(len(compile_times[n]) == NUM_HEAVISIDE_RUNS)
        assert(len(eval_times[n]) == NUM_HEAVISIDE_RUNS)
        size = get_ast_size(dexpr)
        print(f"AST size: {size}")
        ast_sizes.append(size)

    print(num_heavisides)
    print(compile_times)
    print(eval_times)
    print(ast_sizes)
    return num_heavisides, compile_times, eval_times, ast_sizes

