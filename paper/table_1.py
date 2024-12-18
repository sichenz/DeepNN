# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import re
import numpy as np
import pandas as pd

import modules.args
import modules.lib


def load_data(all_path_data):
    _data = []
    for p in all_path_data:
        fp = f"{p}/benchmarking.csv"
        if os.path.isfile(fp):
            res_f = pd.read_csv(fp)
            res_f["r"] = np.int32(re.compile(r".*_([0-9]+)$").findall(p)[0])
            _data.append(res_f)
    data = pd.concat(_data).melt(["model", "r"])
    data_p = data[data["model"] == "p"].rename(columns={"value": "value_p"})
    del data_p["model"]
    data = data.merge(data_p, on=["r", "variable"], how="left")
    data.eval("delta = value - value_p", inplace=True)
    return data


table_models = {
    "p": "True Probabilities",
    "DNN": "Our Model",
    "BinaryLogit_Cross_ByJ": "Binary Logit",
    "LightGBM_Cross_ByJ": "LightGBM",
    "LightGBM_Cat_Cross_ByJ": "LightGBM Category",
    "MXL": "Hierarchical MNL",
}


if __name__ == "__main__":

    TABLE = 1

    # args
    args = modules.args.global_args(f"Table {TABLE} in paper.")
    args.l = "bootstrap"
    all_path_data = modules.lib.get_data_paths(args)

    # load config
    config = modules.lib.read_yaml(args.c)
    path_data = config["path_data"]
    path_results = os.path.expandvars(f"{config['path_results']}/{args.f}")
    os.makedirs(path_results, exist_ok=True)

    # data plot
    df = load_data(all_path_data)
    table_models = {k: table_models[k] for k in table_models if k in df.model.values}
    df["model_print"] = df["model"].map(table_models)
    df = df[df["model_print"].notnull()]
    # df[df["variable"]=="log-loss"].groupby('model').value.mean()#.round(4)

    # result
    mean = (
        df[df["variable"].isin([args.m])]
        .groupby("model_print")["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .round(4)
    )
    delta = (
        df[df["variable"].isin([args.m])]
        .groupby(["model_print"])["delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .round(4)
    )
    res = mean.merge(delta, on="model_print", suffixes=["", "_delta"])
    res["se"] = res["std"] / np.sqrt(res["count"])
    res["se_delta"] = res["std_delta"] / np.sqrt(res["count_delta"])
    res = res[["model_print", "mean", "se", "mean_delta", "se_delta"]].set_index(
        "model_print"
    )
    res = res.loc[table_models.values()]
    res.index.name = "model"
    print(res)
    res.to_html(f"{path_results}/table_{TABLE}_{args.m}.html")
