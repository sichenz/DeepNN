# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import re
import numpy as np
import pandas as pd

import modules.args
import modules.lib


def load_data(paths, labels):
    _results = []
    for l in labels:
        for p in paths:
            file_p = f"{p}/prob_uplift/results/optim_{labels[l]}.csv"
            if os.path.isfile(file_p) and (os.stat(file_p).st_size > 0):
                df = pd.read_csv(file_p)
                df["type"] = l
                df["f"] = os.path.basename(p)
                df0 = df[df["var"].isin(no_discount_map)].reset_index(drop=True)
                df0["model"] = df0["var"].map(no_discount_map)
                df["var"] = df["var"].str.replace(f"{l}_", "")
                model_map_2 = {
                    k: model_map[k] for k in model_map if k in df["var"].values
                }
                df["model"] = df["var"].map(model_map_2)
                df = df.set_index("var").loc[model_map_2.keys()]
                df0[["model", "type", "f", "Total", "Own"]]
                _results.append(
                    df[["model", "type", "f", "Total", "Own"]].reset_index(drop=True)
                )
                _results.append(
                    df0[["model", "type", "f", "Total", "Own"]].reset_index(drop=True)
                )
    return pd.concat(_results).reset_index(drop=True)


def agg_data(x):
    _result = []
    for agg in ["Total", "Own"]:
        res_agg = x.groupby("model")[agg].agg(["mean", "std", "count"])
        res_agg["se"] = res_agg["std"] / np.sqrt(res_agg["count"])
        res_agg[agg] = res_agg.apply(
            lambda x: f"{x['mean']:.2f} ({x['se']:.2f})", axis=1
        )
        _result.append(res_agg[[agg]])
    return pd.concat(_result, axis=1)


if __name__ == "__main__":

    table = 4

    # args
    args = modules.args.global_args("Benchmarking overview table.")
    args.l = "bootstrap"
    all_path_data = modules.lib.get_data_paths(args)

    # load config
    config = modules.lib.read_yaml(args.c)
    path_data = config["path_data"]
    path_results = os.path.expandvars(f"{config['path_results']}/{args.f}")
    os.makedirs(path_results, exist_ok=True)

    # variable maps
    model_map = {
        "true": "True Probabilities",
        "dnn_model_010": "Our Model",
        "logit_cross_by_j": "Binary Logit",
        "lightgbm_cross_by_j": "LightGBM",
        "lightgbm_cat_cross_by_j": "LightGBM Category",
        "mxl_inv": "Hierarchical MNL",
        "uniform-true": "Best Uniform True",
        "random": "Random",
    }
    suffix_map = {
        "prob": "probability",
        "prob_uplift": "probability_uplift",
        "rev": "revenue",
        "rev_uplift": "revenue_uplift",
    }
    no_discount_map = {
        "prob_no_disc_true": "No discount",
        "rev_no_disc_true": "No discount",
    }

    # load data
    tmp = load_data(all_path_data, suffix_map)
    model_map_2 = {
        k: model_map[k] for k in model_map if model_map[k] in tmp["model"].values
    }
    model_list = list(model_map_2.values()) + [list(no_discount_map.values())[0]]
    res_prob = agg_data(tmp[tmp["type"] == "prob_uplift"]).loc[model_map_2.values()]
    print(res_prob)
    res_rev = agg_data(tmp[tmp["type"] == "rev_uplift"]).loc[model_map_2.values()]
    print(res_rev)

    # write output
    res_rev.to_html(f"{path_results}/table_{table}-a-revenue-uplift.html")
    res_prob.to_html(f"{path_results}/table_{table}-b-probability-uplift.html")
