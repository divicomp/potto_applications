import sys
import time
import matplotlib.pyplot as plt
from teg import (
    Const,
    Var,
    TegVar,
    IfElse,
    Teg,
    Tup,
    LetIn,
)
from teg.derivs.fwd_deriv import fwd_deriv
from teg.derivs.reverse_deriv import reverse_deriv
from teg.passes.reduce import reduce_to_base
from teg.eval import evaluate, get_ast_size
from teg.derivs.fwd_deriv import fwd_deriv


sys.setrecursionlimit(10**6)

NUM_RUNS = 10

def run_teg_shader_swap_microbenchmark(num_shader_swap=10, num_samples=10):
    x = TegVar("x")
    a = Var("a")

    def shader1_func(a_, x_):
        return a_ + x_

    def shader2_func(a_, x_):
        return a_ - x_


    # Construct the big expression
    # \sum_i=10^100 H(x + ai)
    # where a_i = 1/i and da_i = 0
    eval_times, compile_times, ast_sizes = [], [], []
    num_shader_swaps = list(range(0, num_shader_swap, 1))
    for n in num_shader_swaps:
        ctx = []
        def do(func):
            big = Const(0)
            for i in range(10, 100):
                new_a = Var(f"a{i}", float(1 / i))
                big += IfElse(func(x, new_a) > 0, 1, 0) * func(new_a, x)
                ctx.append((new_a, 1))
            integrand = big
            integral = Teg(Const(-1), Const(1), integrand, x)
            return integral
        shader1s = []
        for _ in range(n):
            shader1s.append(do(shader1_func))
        shader2 = do(shader2_func)
        expr = shader2
        for shader1 in shader1s:
            expr += shader1

        compile_time_sum = 0.0
        eval_time_sum = 0.0
        for _ in range(NUM_RUNS):
            start = time.time()
            dintegral = fwd_deriv(expr, ctx)
            delta_free = reduce_to_base(dintegral, True)
            compile_time = time.time() - start
            compile_time_sum += compile_time

            start = time.time()
            _ = evaluate(delta_free, backend="numpy", num_samples=num_samples)
            eval_time = time.time() - start
            eval_time_sum += eval_time
        compile_time = compile_time_sum / NUM_RUNS
        print(f'Compile duration: {compile_time}')
        eval_time = eval_time_sum / NUM_RUNS
        print(f'evaluation duration: {eval_time}')
        ast_size = get_ast_size(delta_free)
        print(f"ast size: {ast_size}")
        compile_times.append(compile_time)
        eval_times.append(eval_time)
        ast_sizes.append(ast_size)

    print(num_shader_swaps)
    print(compile_times)
    print(eval_times)
    print(ast_sizes)
    return num_shader_swaps, compile_times, eval_times, ast_sizes


def run_teg_heaviside_microbenchmark(num_heaviside=10, num_samples=10):
    x = TegVar("x")
    a = Var("a")

    def shader1_func(a_, x_):
        return a_ + x_

    def shader2_func(a_, x_):
        return a_ - x_


    # Construct the big expression
    # \sum_i=10^100 H(x + ai)
    # where a_i = 1/i and da_i = 0
    eval_times, compile_times, ast_sizes = [], [], []
    num_heavisides = list(range(0, num_heaviside, 1))
    for n in num_heavisides:
        ctx = []
        def do(func):
            big = Const(0)
            for i in range(10, 10 + n):
                new_a = Var(f"a{i}", float(1 / i))
                big += IfElse(func(x, new_a) > 0, 1, 0) * func(new_a, x)
                ctx.append((new_a, 1))
            integrand = big
            integral = Teg(Const(-1), Const(1), integrand, x)
            return integral
        shader1 = do(shader1_func)
        shader2 = do(shader2_func)
        expr = shader2
        expr += shader1

        compile_time_sum = 0.0
        eval_time_sum = 0.0
        for _ in range(NUM_RUNS):
            start = time.time()
            dintegral = fwd_deriv(expr, ctx)
            delta_free = reduce_to_base(dintegral, True)
            compile_time = time.time() - start
            compile_time_sum += compile_time

            start = time.time()
            _ = evaluate(delta_free, backend="numpy", num_samples=num_samples)
            eval_time = time.time() - start
            eval_time_sum += eval_time
        compile_time = compile_time_sum / NUM_RUNS
        print(f'Compile duration: {compile_time}')
        eval_time = eval_time_sum / NUM_RUNS
        print(f'evaluation duration: {eval_time}')
        ast_size = get_ast_size(delta_free)
        print(f"ast size: {ast_size}")
        compile_times.append(compile_time)
        eval_times.append(eval_time)
        ast_sizes.append(ast_size)

    print(num_heavisides)
    print(compile_times)
    print(eval_times)
    print(ast_sizes)
    return num_heavisides, compile_times, eval_times, ast_sizes

if __name__ == "__main__":
    num_shader_swaps, compile_times, eval_times, ast_sizes = run_teg_shader_swap_microbenchmark(num_shader_swap=4, num_samples=10)
    plt.title("Number of shader swaps microbenchmark")
    plt.plot(num_shader_swaps, compile_times, label="compile time")
    plt.plot(num_shader_swaps, eval_times, label="eval time")
    plt.legend(loc="upper right")
    plt.show()

    num_heavisides, compile_times, eval_times, ast_sizes = run_teg_heaviside_microbenchmark(num_heaviside=4, num_samples=10)
    plt.title("Number of Heavisides microbenchmark")
    plt.plot(num_heavisides, compile_times, label="compile time")
    plt.plot(num_heavisides, eval_times, label="eval time")
    plt.legend(loc="upper right")
    plt.show()
