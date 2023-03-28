import matplotlib.pyplot as plt
import pandas as pd

# Teg data
ast_nodes = "Number of AST Nodes"
df1 = pd.DataFrame(
    {
        "Compile Time": [1375.0, 7191.8],
        "Eval Time": [1115.8, 3613.0],
        "Total Time": [2491.7, 10804.8],
        ast_nodes: [11712, 56511],
    }
)

# Potto data
df2 = pd.DataFrame(
    {
        "Compile Time": [15.6, 16.3],
        "Eval Time": [447.1, 457.1],
        "Total Time": [462.7, 473.4],
        ast_nodes: [3315, 3507],
    }
)

def plot():
    # normalize df2 to df1
    df2_norm = df2 / df1
    df1_norm = df1 / df1

    # plot data
    fig, axarr = plt.subplots(1, len(df1.columns), figsize=(16, 4))

    for i, col in enumerate(df1.columns):
        ax = axarr[i]
        offset = 0.3
        teg_label = "Teg" if col == ast_nodes else None
        potto_label = "Potto" if col == ast_nodes else None
        ax.bar([0, 1], df1_norm[col], width=0.25, align="center", label=teg_label)
        ax.bar([offset, 1 + offset], df2_norm[col], width=0.25, align="center", label=potto_label)
        ax.set_xticks([offset / 2, 1 + offset /2])
        ax.set_xticklabels(["Linear Shader", "Quadratic Shader"])
        # ax.set_ylabel("performance factor", labelpad=10)
        ax.set_title(col)
        ax.set_ylim(0, max(*df1_norm[col], *df2_norm[col]) * 1.1)
        # ax.legend()

        # add labels above bars
        for j, val in enumerate(df1[col]):
            ax.text(j, val + 0.1 * val, str(val), ha="center")

        for j, (df1_val, df1_norm_val, df2_val, df2_norm_val) in enumerate(zip(df1[col], df1_norm[col], df2[col], df2_norm[col])):
            mult = df2_norm_val / df1_norm_val
            units = "" if col == ast_nodes else "ms"
            y_offset = 0.03
            ax.text(j, df1_norm_val + y_offset, f"{df1_val:.0f}{units}", ha="center")
            ax.text(j + offset, df2_norm_val + y_offset, f"{df2_val:.0f}{units}\n{mult:.3f}x", ha="center")
    fig.legend(loc="lower center", ncol=2)
    fig.subplots_adjust(bottom=0.2)
    plt.show()
    # save figure
    # plt.savefig("test_figure.png")

if __name__ == "__main__":
    plot()
