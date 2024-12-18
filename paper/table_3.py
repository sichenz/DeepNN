# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import re
import numpy as np
import pandas as pd

import modules.args
import modules.lib


model_map = {
    "el_true": "True Probabilities",
    "el_dnn_model_010": "Our Model",
    "el_logit_cross_by_j": "Binary Logit",
    "el_lightgbm_cross_by_j": "LightGBM",
    "el_lightgbm_cat_cross_by_j": "LightGBM Category",
    "el_mxl_inv": "Hierarchical MNL",
}


def load_data(paths, label):
    """
    paths = all_path_data
    """
    _results = []
    for p in paths:
        file_p = f"{p}/prob_uplift/results/elasticities_{label}.csv"
        if os.path.isfile(file_p) and (os.stat(file_p).st_size > 0):
            df = pd.read_csv(file_p)
            model_map_2 = {k: model_map[k] for k in model_map if k in df["var"].values}
            df["model"] = df["var"].map(model_map_2)
            df = df.set_index("var").loc[model_map_2.keys()]
            df["f"] = os.path.basename(p)
            _results.append(df[["model", "f", "Mean", "MAE"]].reset_index(drop=True))
    return pd.concat(_results)


def agg_data(x, v, l):
    res_agg = x.groupby("model")[v].agg(["mean", "std", "count"])
    res_agg["se"] = res_agg["std"] / np.sqrt(res_agg["count"])
    res_agg["res"] = res_agg.apply(lambda x: f"{x['mean']:.3f} ({x['se']:.3f})", axis=1)
    res_agg["type"] = label
    res_agg["variable"] = v
    return res_agg[["type", "variable", "res"]]


if __name__ == "__main__":

    TABLE = 3

    # args
    args = modules.args.global_args(f"Table {TABLE} in paper.")
    args.l = "bootstrap"
    all_path_data = modules.lib.get_data_paths(args)

    # load config
    config = modules.lib.read_yaml(args.c)
    path_data = config["path_data"]
    path_results = os.path.expandvars(f"{config['path_results']}/{args.f}")
    os.makedirs(path_results, exist_ok=True)

    all_labels = ["own", "within", "cross_pos", "cross_neg", "cross_0"]
    _res = []
    for label in all_labels:
        tmp = load_data(all_path_data, label)
        _res.append(agg_data(tmp, "Mean", label).reset_index())
        _res.append(agg_data(tmp, "MAE", label).reset_index())
    res = pd.concat(_res).reset_index(drop=True)
    model_map_2 = {
        k: model_map[k] for k in model_map if model_map[k] in res["model"].values
    }
    res_mean = (
        res[res["variable"] == "Mean"]
        .pivot_table("res", "model", "type", aggfunc=lambda x: " ".join(x))
        .loc[model_map_2.values()][all_labels]
    )
    print(res_mean)
    res_mae = (
        res[res["variable"] == "MAE"]
        .pivot_table("res", "model", "type", aggfunc=lambda x: " ".join(x))
        .loc[model_map_2.values()][all_labels]
    )
    print(res_mae)
    res_mean.to_html(f"{path_results}/table_{TABLE}-a.html")
    res_mae.to_html(f"{path_results}/table_{TABLE}-b.html")
