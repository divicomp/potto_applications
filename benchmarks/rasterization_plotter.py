import matplotlib.pyplot as plt
import pandas as pd

# Teg data
ast_nodes = "Number of AST Nodes"


def plot():
    df1 = pd.DataFrame(
        {
            "Compile Time": [1375.0, 7191.8],
            "Evaluation Time": [1115.8, 3613.0],
            "Total Time": [2491.7, 10804.8],
            ast_nodes: [11712, 56511],
        }
    )

    # Potto data
    df2 = pd.DataFrame(
        {
            "Compile Time": [15.6, 16.3],
            "Evaluation Time": [447.1, 457.1],
            "Total Time": [462.7, 473.4],
            ast_nodes: [3315, 3507],
        }
    )

    fig, axarr = plt.subplots(1, len(df1.columns), figsize=(16, 4))

    for i, col in enumerate(df1.columns):
        if i < 3:
            df1[col] = df1[col] / 1000
            df2[col] = df2[col] / 1000

    for i, col in enumerate(df1.columns):
        ax = axarr[i]
        offset = 0.3
        teg_label = "Teg" if col == ast_nodes else None
        potto_label = "Potto" if col == ast_nodes else None
        ax.bar([0, 1], df2[col], width=0.25, align="center", label=potto_label, color="#FE6100")
        ax.bar([offset, 1 + offset], df1[col], width=0.25, align="center", label=teg_label, color="#648FFF")
        ax.set_xticks([offset / 2, 1 + offset / 2])
        ax.set_xticklabels(["Linear Shader", "Quadratic Shader"], fontsize=12)
        if i < 3:
            ax.set_ylabel("Time (s)", labelpad=5, fontsize=12)
            max_ylim = 0
            for c in df2.columns[:3]:
                max_ylim = max(*df1[c], *df2[c], max_ylim) * 1.15
            ax.set_ylim(0, max_ylim)
        else:
            ax.set_ylabel("# AST Nodes", labelpad=5, fontsize=12)
            ax.set_ylim(0, max(*df1[col], *df2[col]) * 1.15)
        ax.set_title(col, fontsize=18)

        for j, (df1_val, df2_val) in enumerate(zip(df1[col], df2[col])):
            mult = df1_val / df2_val
            # y_offset = 0.03 * max(*df1_norm[col], *df2_norm[col])
            y_offset = 0.02 * max_ylim if i < 3 else 0.02 * max(*df1[col], *df2[col])
            df1y = df1_val + y_offset
            ax.text(j + offset, df1y, f"{mult:.1f}x", ha="center")
    fig.legend(loc="lower center", ncol=2, fontsize=14)
    fig.subplots_adjust(bottom=0.2, left=0.05, right=0.95, wspace=0.4)
    plt.show()
    # save figure
    # plt.savefig("test_figure.png")

if __name__ == "__main__":
    plot()
