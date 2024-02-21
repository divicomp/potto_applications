import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.pyplot import figure
import numpy as np
import argparse
import os

DATA_DIR_PREFIX = "benchmark_data/"


def geo_mean_overflow(iterable):
    return np.exp(np.log(iterable).mean())


def plot_heavisides(num_heaviside=10, load_local=False, save_plot=False, path=None):
    if load_local:
        teg_num_heavisides = np.load(DATA_DIR_PREFIX + "teg_num_heavisides.npy")
        teg_compile_times = np.load(DATA_DIR_PREFIX + "teg_compile_times_heavisides.npy")
        teg_eval_times = np.load(DATA_DIR_PREFIX + "teg_eval_times_heavisides.npy")
        teg_total_times = np.load(DATA_DIR_PREFIX + "teg_total_times_heavisides.npy")
        teg_ast_sizes = np.load(DATA_DIR_PREFIX + "teg_ast_sizes_heavisides.npy")

        potto_num_heavisides = np.load(DATA_DIR_PREFIX + "potto_num_heavisides.npy")
        potto_compile_times = np.load(DATA_DIR_PREFIX + "potto_compile_times_heavisides.npy")
        potto_eval_times = np.load(DATA_DIR_PREFIX + "potto_eval_times_heavisides.npy")
        potto_total_times = np.load(DATA_DIR_PREFIX + "potto_total_times_heavisides.npy")
        potto_ast_sizes = np.load(DATA_DIR_PREFIX + "potto_ast_sizes_heavisides.npy")
    else:
        from teg_microbenchmarks import run_teg_heaviside_microbenchmark
        from potto_microbenchmarks import run_potto_heaviside_microbenchmark

        teg_num_heavisides, teg_compile_times, teg_eval_times, teg_ast_sizes = run_teg_heaviside_microbenchmark(
            num_heaviside=num_heaviside, num_samples=10
        )
        potto_num_heavisides, potto_compile_times, potto_eval_times, potto_ast_sizes = (
            run_potto_heaviside_microbenchmark(num_heaviside=num_heaviside, num_samples=10)
        )
        teg_total_times = np.array(teg_compile_times) + np.array(teg_eval_times)
        potto_total_times = np.array(potto_compile_times) + np.array(potto_eval_times)

        np.save(DATA_DIR_PREFIX + "teg_num_heavisides.npy", np.array(teg_num_heavisides))
        np.save(DATA_DIR_PREFIX + "teg_compile_times_heavisides.npy", np.array(teg_compile_times))
        np.save(DATA_DIR_PREFIX + "teg_eval_times_heavisides.npy", np.array(teg_eval_times))
        np.save(DATA_DIR_PREFIX + "teg_total_times_heavisides.npy", np.array(teg_total_times))
        np.save(DATA_DIR_PREFIX + "teg_ast_sizes_heavisides.npy", np.array(teg_ast_sizes))

        np.save(DATA_DIR_PREFIX + "potto_num_heavisides.npy", np.array(potto_num_heavisides))
        np.save(DATA_DIR_PREFIX + "potto_compile_times_heavisides.npy", np.array(potto_compile_times))
        np.save(DATA_DIR_PREFIX + "potto_eval_times_heavisides.npy", np.array(potto_eval_times))
        np.save(DATA_DIR_PREFIX + "potto_total_times_heavisides.npy", np.array(potto_total_times))
        np.save(DATA_DIR_PREFIX + "potto_ast_sizes_heavisides.npy", np.array(potto_ast_sizes))

    teg_compile_times_mean = np.array([geo_mean_overflow(x) for x in teg_compile_times])
    potto_compile_times_mean = np.array([geo_mean_overflow(x) for x in potto_compile_times])
    teg_eval_times_mean = np.array([geo_mean_overflow(x) for x in teg_eval_times])
    potto_eval_times_mean = np.array([geo_mean_overflow(x) for x in potto_eval_times])
    teg_total_times_mean = np.array([geo_mean_overflow(x) for x in teg_total_times])
    potto_total_times_mean = np.array([geo_mean_overflow(x) for x in potto_total_times])

    fs = 16
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    axs[0].set_xlabel("Number of Deltas", fontsize=fs)
    axs[0].set_ylabel("Time(s)", fontsize=fs)
    axs[0].set_yscale("log")
    axs[0].set_title("Compile Time", fontsize=fs)
    axs[0].set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 0])
    axs[0].minorticks_off()
    axs[0].plot(potto_num_heavisides, potto_compile_times_mean, label="Potto", linewidth=4, color="#FE6100")
    axs[0].plot(teg_num_heavisides, teg_compile_times_mean, label="Teg", linewidth=4, color="#648FFF")

    axs[1].set_xlabel("Number of Deltas", fontsize=fs)
    axs[1].set_ylabel("Time(s)", fontsize=fs)
    axs[1].set_yscale("log")
    axs[1].set_title("Evaluation Time", fontsize=fs)
    axs[1].minorticks_off()
    axs[1].plot(potto_num_heavisides, potto_eval_times_mean, label="Potto", linewidth=4, color="#FE6100")
    axs[1].plot(teg_num_heavisides, teg_eval_times_mean, label="Teg", linewidth=4, color="#648FFF")

    axs[2].set_xlabel("Number of Deltas", fontsize=fs)
    axs[2].set_ylabel("Time(s)", fontsize=fs)
    axs[2].set_yscale("log")
    axs[2].set_title("Total Time", fontsize=fs)
    axs[2].minorticks_off()
    axs[2].plot(potto_num_heavisides, potto_total_times_mean, label="Potto", linewidth=4, color="#FE6100")
    axs[2].plot(teg_num_heavisides, teg_total_times_mean, label="Teg", linewidth=4, color="#648FFF")

    axs[3].set_xlabel("Number of Deltas", fontsize=fs)
    axs[3].set_ylabel("# AST Node", fontsize=fs)
    axs[3].set_title("Number of AST Nodes", fontsize=fs)
    axs[3].plot(potto_num_heavisides, potto_ast_sizes, label="Potto", linewidth=4, color="#FE6100")
    axs[3].plot(teg_num_heavisides, teg_ast_sizes, label="Teg", linewidth=4, color="#648FFF")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.15), loc="lower center", ncol=2)
    fig.tight_layout()
    if save_plot:
        plt.savefig(f"{path}/heavisides.svg", bbox_inches="tight")
        plt.clf()
    else:
        plt.show()


def plot_shader_swaps(num_shader_swap=10, load_local=False, save_plot=False, path=None):
    if load_local:
        teg_num_shader_swaps = np.load(DATA_DIR_PREFIX + "teg_num_shader_swaps.npy")
        teg_compile_times = np.load(DATA_DIR_PREFIX + "teg_compile_times_shader_swaps.npy")
        teg_eval_times = np.load(DATA_DIR_PREFIX + "teg_eval_times_shader_swaps.npy")
        teg_total_times = np.load(DATA_DIR_PREFIX + "teg_total_times_shader_swaps.npy")
        teg_ast_sizes = np.load(DATA_DIR_PREFIX + "teg_ast_sizes_shader_swaps.npy")

        potto_num_shader_swaps = np.load(DATA_DIR_PREFIX + "potto_num_shader_swaps.npy")
        potto_compile_times = np.load(DATA_DIR_PREFIX + "potto_compile_times_shader_swaps.npy")
        potto_eval_times = np.load(DATA_DIR_PREFIX + "potto_eval_times_shader_swaps.npy")
        potto_total_times = np.load(DATA_DIR_PREFIX + "potto_total_times_shader_swaps.npy")
        potto_ast_sizes = np.load(DATA_DIR_PREFIX + "potto_ast_sizes_shader_swaps.npy")
    else:
        from teg_microbenchmarks import run_teg_shader_swap_microbenchmark
        from potto_microbenchmarks import run_potto_shader_swap_microbenchmark

        potto_num_shader_swaps, potto_compile_times, potto_eval_times, potto_ast_sizes = (
            run_potto_shader_swap_microbenchmark(num_shader_swap=num_shader_swap, num_samples=10)
        )
        potto_total_times = np.array(potto_compile_times) + np.array(potto_eval_times)
        np.save(DATA_DIR_PREFIX + "potto_num_shader_swaps.npy", np.array(potto_num_shader_swaps))
        np.save(DATA_DIR_PREFIX + "potto_compile_times_shader_swaps.npy", np.array(potto_compile_times))
        np.save(DATA_DIR_PREFIX + "potto_eval_times_shader_swaps.npy", np.array(potto_eval_times))
        np.save(DATA_DIR_PREFIX + "potto_total_times_shader_swaps.npy", np.array(potto_total_times))
        np.save(DATA_DIR_PREFIX + "potto_ast_sizes_shader_swaps.npy", np.array(potto_ast_sizes))

        teg_num_shader_swaps, teg_compile_times, teg_eval_times, teg_ast_sizes = run_teg_shader_swap_microbenchmark(
            num_shader_swap=min(num_shader_swap, 15), num_samples=10
        )
        teg_total_times = np.array(teg_compile_times) + np.array(teg_eval_times)
        np.save(DATA_DIR_PREFIX + "teg_num_shader_swaps.npy", np.array(teg_num_shader_swaps))
        np.save(DATA_DIR_PREFIX + "teg_compile_times_shader_swaps.npy", np.array(teg_compile_times))
        np.save(DATA_DIR_PREFIX + "teg_eval_times_shader_swaps.npy", np.array(teg_eval_times))
        np.save(DATA_DIR_PREFIX + "teg_total_times_shader_swaps.npy", np.array(teg_total_times))
        np.save(DATA_DIR_PREFIX + "teg_ast_sizes_shader_swaps.npy", np.array(teg_ast_sizes))

    teg_compile_times_mean = np.array([geo_mean_overflow(x) for x in teg_compile_times])
    potto_compile_times_mean = np.array([geo_mean_overflow(x) for x in potto_compile_times])
    teg_eval_times_mean = np.array([geo_mean_overflow(x) for x in teg_eval_times])
    potto_eval_times_mean = np.array([geo_mean_overflow(x) for x in potto_eval_times])
    teg_total_times_mean = np.array([geo_mean_overflow(x) for x in teg_total_times])
    potto_total_times_mean = np.array([geo_mean_overflow(x) for x in potto_total_times])

    fs = 16
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    axs[0].set_xlabel("Number of Shader Swaps", fontsize=fs)
    axs[0].set_ylabel("Time(s)", fontsize=fs)
    axs[0].set_yscale("log")
    axs[0].set_title("Compile Time", fontsize=fs)
    axs[0].set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 0])
    axs[0].minorticks_off()
    axs[0].plot(potto_num_shader_swaps, potto_compile_times_mean, label="Potto", linewidth=4, color="#FE6100")
    axs[0].plot(teg_num_shader_swaps, teg_compile_times_mean, label="Teg", linewidth=4, color="#648FFF")

    axs[1].set_xlabel("Number of Shader Swaps", fontsize=fs)
    axs[1].set_ylabel("Time(s)", fontsize=fs)
    axs[1].set_yscale("log")
    axs[1].set_title("Evaluation Time", fontsize=fs)
    axs[1].minorticks_off()
    axs[1].plot(potto_num_shader_swaps, potto_eval_times_mean, label="Potto", linewidth=4, color="#FE6100")
    axs[1].plot(teg_num_shader_swaps, teg_eval_times_mean, label="Teg", linewidth=4, color="#648FFF")

    axs[2].set_xlabel("Number of Shader Swaps", fontsize=fs)
    axs[2].set_ylabel("Time(s)", fontsize=fs)
    axs[2].set_yscale("log")
    axs[2].set_title("Total Time", fontsize=fs)
    axs[2].minorticks_off()
    axs[2].plot(potto_num_shader_swaps, potto_total_times_mean, label="Potto", linewidth=4, color="#FE6100")
    axs[2].plot(teg_num_shader_swaps, teg_total_times_mean, label="Teg", linewidth=4, color="#648FFF")

    axs[3].set_xlabel("Number of Shader Swaps", fontsize=fs)
    axs[3].set_ylabel("# AST Node", fontsize=fs)
    axs[3].set_title("Number of AST Nodes", fontsize=fs)
    axs[3].plot(potto_num_shader_swaps, potto_ast_sizes, label="Potto", linewidth=4, color="#FE6100")
    axs[3].plot(teg_num_shader_swaps, teg_ast_sizes, label="Teg", linewidth=4, color="#648FFF")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.15), loc="lower center", ncol=2)
    fig.tight_layout()
    if save_plot:
        plt.savefig(f"{path}/shader_swaps.svg", bbox_inches="tight")
        plt.clf()
    else:
        plt.show()
    # plt.show()
    # plt.savefig("shader_swaps.png", bbox_inches = "tight")


if __name__ == "__main__":
    path = "/Users/jessemichel/research/potto_project/potto_paper/images/benchmarks"
    save = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="Load local results from previous run")
    parser.add_argument("--heaviside", action="store_true", help="Run heaviside microbenchmark")
    parser.add_argument("--shader-swap", action="store_true", help="Run shader swap microbenchmark")
    parser.add_argument("--all", action="store_true", help="Run all microbenchmarks")
    args = parser.parse_args()
    if not os.path.exists(DATA_DIR_PREFIX):
        os.makedirs(DATA_DIR_PREFIX)
    if args.heaviside or args.all:
        print("run heaviside")
        plot_heavisides(50, args.load, save_plot=False, path=path)
    if args.shader_swap or args.all:
        print("run shader swap")
        plot_shader_swaps(50, args.load, save, path)
