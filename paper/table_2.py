# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import re
import numpy as np
import pandas as pd

import modules.args
import modules.lib
import paper.table_1


if __name__ == "__main__":

    TABLE = 2

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
    df = paper.table_1.load_data(all_path_data)
    table_models = paper.table_1.table_models
    table_models = {k: table_models[k] for k in table_models if k in df.model.values}
    df["model_print"] = df["model"].map(table_models)
    df = df[df["model_print"].notnull()]

    # result
    mean = (
        df[df["variable"].isin(["time-correlation"])]
        .groupby("model_print")["value"]
        .agg(["mean", "std"])
        .reset_index()
        .round(2)
    )
    mean_no_discount = (
        df[df["variable"].isin(["time-correlation_no_discount"])]
        .groupby("model_print")["value"]
        .agg(["mean", "std"])
        .reset_index()
        .round(2)
    )
    res = mean.merge(mean_no_discount, on="model_print", suffixes=["", "_no_discount"])
    res["se"] = res["std"] / np.sqrt(df["r"].nunique())
    res["se_no_discount"] = res["std_no_discount"] / np.sqrt(df["r"].nunique())
    res = res[
        ["model_print", "mean", "se", "mean_no_discount", "se_no_discount"]
    ].set_index("model_print")
    res = res.loc[table_models.values()]
    res.index.name = "model"
    print(res)
    res.to_html(f"{path_results}/table_{TABLE}.html")
