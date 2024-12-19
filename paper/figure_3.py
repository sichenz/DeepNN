# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import torch
import sklearn.manifold
import scipy.stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

import modules.args
import modules.lib

matplotlib.rcParams['font.family'] = 'Iowan Old Style'
sns.set(font="DejaVu Sans")

if __name__ == "__main__":

    args = modules.args.global_args("Figure 3")
    config = modules.lib.read_yaml(args.c)
    path_data = os.path.expandvars(f"{config['path_data']}_000")
    path_results = os.path.expandvars(f"{config['path_results']}/paper")
    os.makedirs(path_results, exist_ok=True)

    data_set = 2
    epoch = 99
    eps = 1.0e-3
    R = 20
    seed = 501
    figure = "3"

    data_j = pd.read_csv(f"{path_data}/data_j.csv")
    sd = torch.load(
        f"{path_data}/model_010/results/state_dict_{epoch:08d}.pt",
        map_location=torch.device("cpu"),
        weights_only=True
    )

    plt.rcParams.update(**config["figure_3"]["rc"])
    
    # Set font after updating rcParams and Seaborn settings
    matplotlib.rcParams['font.family'] = 'Iowan Old Style'
    sns.set(font="Iowan Old Style")

    for var in ["w_conv_j", "w_conv_j_d"]:

        conv_j = sd[var].cpu().numpy()
        conv_j_norm = conv_j / np.linalg.norm(conv_j, axis=1)[:, np.newaxis]

        J = conv_j.shape[0]
        K = conv_j.shape[1]

        p = plt.figure(figsize=(8.00, 5.33))

        ax = sns.heatmap(pd.DataFrame(conv_j_norm), cmap="YlGnBu", vmin=-0.5, vmax=0.5)
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Product")
        plt.tight_layout()
        plt.xlim(0, K)
        plt.ylim(0, J)
        ax.set_yticks(np.arange(0, 251, 10))
        ax.set_yticklabels(np.arange(0, 251, 10))
        ax.set_xticks(np.arange(0, 31, 2))
        ax.set_xticklabels(np.arange(0, 31, 2))
        p.savefig(f"{path_results}/figure-{figure}-a-{var}.png")

        # tsne
        np.random.seed(seed)
        min_kl = 10_000
        for r in range(R):
            tsne = sklearn.manifold.TSNE(**config["tsne"]).fit(conv_j_norm)
            if tsne.kl_divergence_ < min_kl:
                min_kl = tsne.kl_divergence_
                xy = tsne.embedding_

        fig, ax = plt.subplots(figsize=(8.00, 6.67))
        fig.tight_layout()
        df_plot = pd.DataFrame(
            {
                "x": scipy.stats.zscore(xy[:, 0])
                + np.random.uniform(-eps, eps, xy.shape[0]),
                "y": scipy.stats.zscore(xy[:, 1])
                + np.random.uniform(-eps, eps, xy.shape[0]),
                "c": data_j["c"].values,
            }
        )
        _ = sns.scatterplot(
            x="x",
            y="y",
            hue="c",
            data=df_plot,
            palette="YlGnBu",
            alpha=0.7,
            legend=False,
            s=150,
        )
        fig.savefig(f"{path_results}/figure-{figure}-b-{var}.png")

        plt.close("all")
