# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches
import matplotlib.pyplot as plt
import modules.args
import modules.lib


if __name__ == "__main__":

    # setup
    args = modules.args.global_args("Figure 2 (Plot)")
    config = modules.lib.read_yaml(args.c)
    plt.rcParams.update(**config["figure_2"]["rc"])
    path_data = os.path.expandvars(f"{config['path_data']}_009")
    path_out = f"{path_data}/figure-2"
    # skip figure 2 if data set 9 is not available
    if not os.path.isdir(path_data):
        exit()

    # manual input
    i_own = 1745
    j_own = 68
    j_within = 64
    j_cross = 222
    t_coupon = 96
    discounts_own = [t_coupon]
    purchase_own = [93]

    # load data
    prediction_master = pd.read_parquet(f"{path_out}/prediction_master_2.parquet")
    baskets = pd.read_parquet(f"{path_data}/baskets.parquet")
    # skip figure 2 if simulation config does not match paper
    if not baskets["i"].nunique() == 100_000:
        exit()
    dnn = pd.read_parquet(
        f"{path_out}/pred_dnn.parquet", columns=["i", "j", "t", "phat"]
    )
    logit = pd.read_parquet(
        f"{path_out}/pred_logit.parquet", columns=["i", "j", "t", "phat"]
    )
    lightgbm = pd.read_parquet(
        f"{path_out}/pred_lightgbm.parquet", columns=["i", "j", "t", "phat"]
    )
    mxl = pd.read_parquet(f"{path_out}/pred_mxl.parquet", columns=['i','j','t','phat'])
    df = (
        prediction_master.merge(dnn.rename(columns={"phat": "dnn"}), on=["i", "j", "t"])
        .merge(logit.rename(columns={"phat": "logit"}), on=["i", "j", "t"])
        .merge(lightgbm.rename(columns={"phat": "gbm"}), on=["i", "j", "t"])
        .merge(mxl.rename(columns={'phat':'mxl'}), on=['i','j','t'])
    )
    df_key = (
        df[["i", "j", "t", "c", "p", "dnn", "logit", "gbm", "mxl"]]
        .reset_index(drop=True)
        .set_index(["i", "j"])
    )

    # plot
    d = {
        "p": {
            "color": sns.color_palette("RdBu_r", 7)[6],
            "lw": 2.1,
            "zorder": 3,
        },
        "dnn": {
            "color": sns.color_palette("RdBu_r", 7)[0],
            "lw": 2,
            "dashes": [8, 1, 8, 1],
            "zorder": 2,
        },
        "logit": {
            "color": "grey",
            "lw": 1,
            "dashes": [1, 1, 1, 1],
            "zorder": 0.8,
        },
        "gbm": {
            "color": "grey",
            "lw": 1,
            "dashes": [8, 2, 2, 2],
            "zorder": 0.8,
        },
        "mxl": {
            "color": "grey",
            "lw": 1,
            "dashes": [4, 2, 4, 2],
            "zorder": 0.8,
        },
    }

    def plot_show(df_key, i, j):
        model_plot = [
            "p",
            "dnn",
            "logit",
            "gbm",
            "mxl",
        ]
        data_plot = df_key.loc[(i, j), :].reset_index().melt(["i", "j", "t", "c"])
        name_map_j = {jx: f"Product {ix+1}" for ix, jx in enumerate(j)}
        name_map_j[j[0]] += " (Coupon)"
        name_map_j[j[1]] += " (Within)"
        name_map_j[j[2]] += " (Cross)"
        data_plot["j_plot"] = data_plot.j.map(name_map_j)
        data_plot["model_order"] = data_plot["variable"].map({"p": 0, "dnn": 1})
        data_plot = data_plot.sort_values(["j_plot", "model_order"])

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.7))
        fig.subplots_adjust(bottom=0.3)
        n_ticks = 4

        # Own-Coupon
        j = "Product 1 (Coupon)"
        ax = axes[0]
        ax.set_title("Product 1 (Focal)")
        data_i = data_plot[data_plot["j_plot"] == j]
        for m in model_plot:
            data_im = data_i[data_i["variable"] == m]
            ax.plot(data_im["t"].values, data_im["value"].values, **d[m])
            ax.grid(True, color=(0.8, 0.8, 0.8))
        for disc in discounts_own:
            ax.axvspan(
                disc - 0.2,
                disc + 0.2,
                alpha=0.5,
                color=sns.color_palette("BrBG", 10)[6],
                lw=0,
            )
        for purch in purchase_own:
            ax.axvspan(
                purch - 0.2,
                purch + 0.2,
                alpha=0.5,
                color=sns.color_palette("coolwarm", 7)[4],
                lw=0,
            )
        ax.set_xlabel("Week")
        ax.set_ylabel("Purchase Probability")
        y_top = data_i["value"].values.max() * 1.05
        y_step = y_top / n_ticks
        mul = np.power(10, np.floor(np.log10(abs(y_step))))
        y_step = np.ceil(y_step / mul) * mul
        ax.set_ylim(0.00, n_ticks * y_step)
        ax.yaxis.set_ticks(np.arange(0.00, (n_ticks + 1) * y_step, y_step))

        # Within-Coupon
        j = "Product 2 (Within)"
        ax = axes[1]
        ax.set_title("Product 2 (Within-Category)")
        data_i = data_plot[data_plot["j_plot"] == j]
        for m in model_plot:
            data_im = data_i[data_i["variable"] == m]
            ax.plot(data_im["t"].values, data_im["value"].values, **d[m])
            ax.grid(True, color=(0.8, 0.8, 0.8))
        for disc in discounts_own:
            ax.axvspan(
                disc - 0.2,
                disc + 0.2,
                alpha=0.5,
                color=sns.color_palette("BrBG", 10)[6],
                lw=0,
            )
        for purch in purchase_own:
            ax.axvspan(
                purch - 0.2,
                purch + 0.2,
                alpha=0.5,
                color=sns.color_palette("coolwarm", 7)[4],
                lw=0,
            )
        ax.set_xlabel("Week")
        y_top = data_i["value"].values.max() * 1.05
        y_step = y_top / n_ticks
        mul = np.power(10, np.floor(np.log10(abs(y_step))))
        y_step = 0.06
        ax.set_ylim(0.00, n_ticks * y_step)
        ax.yaxis.set_ticks(np.arange(0.00, (n_ticks + 1) * y_step, y_step))

        # Cross-Coupon
        j = "Product 3 (Cross)"
        ax = axes[2]
        ax.set_title("Product 3 (Cross-Category)")
        data_i = data_plot[data_plot["j_plot"] == j]
        for m in model_plot:
            data_im = data_i[data_i["variable"] == m]
            ax.plot(data_im["t"].values, data_im["value"].values, **d[m])
            ax.grid(True, color=(0.8, 0.8, 0.8))
        for disc in discounts_own:
            # ax.axvline(x=disc, color='green', zorder=1)
            ax.axvspan(
                disc - 0.2,
                disc + 0.2,
                alpha=0.5,
                color=sns.color_palette("BrBG", 10)[6],
                lw=0,
            )
        for purch in purchase_own:
            ax.axvspan(
                purch - 0.2,
                purch + 0.2,
                alpha=0.5,
                color=sns.color_palette("coolwarm", 7)[4],
                lw=0,
            )
        ax.set_xlabel("Week")
        y_top = data_i["value"].values.max() * 1.05
        y_step = y_top / n_ticks
        mul = np.power(10, np.floor(np.log10(abs(y_step))))
        y_step = np.ceil(y_step / mul) * mul
        ax.set_ylim(0.00, n_ticks * y_step)
        ax.yaxis.set_ticks(np.arange(0.00, (n_ticks + 1) * y_step, y_step))

        # legend
        legend_elements = [
            matplotlib.lines.Line2D([0], [0], **d["p"], label="True Probability"),
            matplotlib.lines.Line2D([0], [0], **d["dnn"], label="Our Model"),
            matplotlib.lines.Line2D([0], [0], color="white", label=""),
            matplotlib.lines.Line2D([0], [0], **d["logit"], label="Binary Logit"),
            matplotlib.lines.Line2D([0], [0], **d["gbm"], label="LightGBM"),
            matplotlib.lines.Line2D([0], [0], **d["mxl"], label="Hierarchical MNL"),
            matplotlib.patches.Patch(
                facecolor=sns.color_palette("coolwarm", 7)[4],
                label="Purchase in Focal Category",
            ),
            matplotlib.patches.Patch(
                facecolor=sns.color_palette("BrBG", 10)[6],
                label="Discount for Focal Product",
            ),
        ]
        fig.legend(
            handles=legend_elements,
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 0),
        )

        return fig

    fig = plot_show(df_key, i_own, [j_own, j_within, j_cross])
    fig.savefig(f"{path_out}/figure_2.png", dpi=250)
    plt.close()
