import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import truncnorm

path = "/Users/jessemichel/research/potto_project/potto_paper/images/"

# def normal_vs_trunc_normal():
#     x = np.arange(-2, 6, 0.001)

#     # plot normal distribution with mean 0 and standard deviation 1
#     plt.plot(
#         x,
#         norm.pdf(x, 2, 5),
#         linewidth=4,
#         color="gray",
#         label="Normal",
#     )
#     plt.plot(
#         x,
#         truncnorm.pdf(x, -2, 3, loc=1, scale=5),
#         linewidth=4,
#         color="green",
#         label="Trunc. normal",
#         linestyle=":",
#     )
#     plt.title("Initialization")
#     # -1, 4, 1, 1
#     plt.legend()
#     plt.show()

width = 7
mean = 2
x = np.arange(mean - width, mean + width, 0.001)


def normal_vs_trunc_normal():
    # plot normal distribution with mean 0 and standard deviation 1
    plt.plot(
        x,
        norm.pdf(x, mean, 5),
        linewidth=4,
        color="gray",
        label="Normal",
    )
    color = "#2b572c"
    plt.plot(
        x,
        truncnorm.pdf(x, -2 / 5, 3 / 5, loc=1, scale=5),
        linewidth=4,
        color=color,
        label="Trunc. normal",
        linestyle=":",
    )
    # plt.title("Initialization", fontsize=25)

    # Add textboxes with LaTeX math
    plt.text(6, 0.063, r"$N(x;2,5)$", fontsize=12, color="gray")
    plt.text(-5.3, 0.15, r"$T(x;-2,3,1,5)$", fontsize=12, color=color)

    plt.xlabel(xlabel="x", fontsize=12)
    plt.ylabel(ylabel="Density", fontsize=12)
    plt.legend()
    # plt.show()
    plt.savefig(path + "normal_vs_trunc.svg")
    plt.clf()


def potto_normal_vs_trunc_normal():
    # plot normal distribution with mean 0 and standard deviation 1
    plt.plot(
        x,
        norm.pdf(x, mean, 5),
        linewidth=4,
        color="gray",
        label="Normal",
    )
    color = "#FE6100"
    plt.plot(
        x,
        truncnorm.pdf(x, (-8.7 - 2) / 5, (12.7 - 2) / 5, loc=2, scale=5),  # -8.7, 12.7, 2
        linewidth=4,
        color=color,
        label="Trunc. normal",
        linestyle=":",
    )
    plt.text(2.8, 0.063, r"$N(x;2,5)$", fontsize=12, color="gray")
    plt.text(-5.3, 0.077, r"$T(x;-8.7,12.7,2,5)$", fontsize=12, color=color)

    # plt.title("Potto", fontsize=25)
    # -1, 4, 1, 1
    plt.xlabel(xlabel="x", fontsize=12)
    plt.ylabel(ylabel="Density", fontsize=12)
    plt.legend()

    # plt.show()
    plt.savefig(path + "potto_normal_vs_trunc.svg")
    plt.clf()


def naivead_normal_vs_trunc_normal():
    # plot normal distribution with mean 0 and standard deviation 1
    plt.plot(
        x,
        norm.pdf(x, mean, 5),
        linewidth=4,
        color="gray",
        label="Normal",
    )
    color = "#6e78ff"
    plt.plot(
        x,
        truncnorm.pdf(x, -3 / 5, 2 / 5, loc=2, scale=5),  # -8.7, 12.7, 2
        linewidth=4,
        color=color,
        label="Trunc. normal",
        linestyle=":",
    )
    plt.text(6, 0.063, r"$N(x;2,5)$", fontsize=12, color="gray")
    plt.text(-5.3, 0.15, r"$T(x;-2,3,2,5)$", fontsize=12, color=color)

    # -1, 4, 1, 5
    # plt.title("Naive AD", fontsize=25)

    plt.xlabel(xlabel="x", fontsize=12)
    plt.ylabel(ylabel="Density", fontsize=12)
    plt.legend()
    # plt.show()
    plt.savefig(path + "naivead_normal_vs_trunc.svg")
    plt.clf()


if __name__ == "__main__":
    normal_vs_trunc_normal()
    potto_normal_vs_trunc_normal()
    naivead_normal_vs_trunc_normal()
