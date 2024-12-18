# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import re

import modules.args
import modules.lib
import numpy as np
import pandas as pd
import sklearn.metrics
import tqdm
from joblib import Parallel, delayed
from loguru import logger


def score_models_rx(r, model_map, df):
    _scores_list = []
    for m in model_map:
        np.random.seed(r)
        idx_r = np.random.choice(range(df.shape[0]), df.shape[0], replace=True)
        df_r = df.iloc[idx_r]
        _scores_list.append(
            pd.DataFrame(
                {
                    "model": [m],
                    "r": r,
                    "log-loss": sklearn.metrics.log_loss(
                        df_r["y"].values, df_r[m].values
                    ),
                    "auc": sklearn.metrics.roc_auc_score(
                        df_r["y"].values, df_r[m].values
                    ),
                }
            )
        )
    return pd.concat(_scores_list)


if __name__ == "__main__":

    # args
    args = modules.args.global_args("TABLE 6")
    args.c = "configs/config-empirical.yaml"
    config = modules.lib.read_yaml(args.c)
    path_data = os.path.expanduser(config["path_data"])
    path_results = os.path.expanduser(config["path_results"])
    logger.info(f"path_data = {path_data}")
    logger.info(f"path_results = {path_results}")
    model = "model_010"
    R = 30
    os.makedirs(path_results, exist_ok=True)

    # variable maps
    model_map = {
        "dnn": "Our Model",
        "freq_product": "Best Uniform",
        "logit": "Binary Logit",
        "lightgbm": "LightGBM",
        "mxl": "Hierarchical MNL",
        "mnl": "MNL",
    }

    prediction_index = pd.read_parquet(
        f"{path_data}/prediction_index.parquet",
        columns=["i", "j", "t", "y"],
    )

    actions = pd.read_parquet(
        f"{path_data}/action.parquet",
        columns=["i", "j", "t", "discount"],
    )

    freq_product = pd.read_parquet(
        f"{path_results}/frequency_product.parquet",
        columns=["i", "j", "t", "phat"],
    ).rename(columns={"phat": "freq_product"})

    logit = pd.read_parquet(
        f"{path_results}/BinaryLogit_Cross_ByJ.parquet",
        columns=["i", "j", "t", "phat"],
    ).rename(columns={"phat": "logit"})

    mxl = pd.read_parquet(
        f"{path_results}/mxl-2k.parquet",
        columns=["i", "j", "t", "phat"],
    ).rename(columns={"phat": "mxl"})

    mnl = pd.read_parquet(
        f"{path_results}/mnl-2k.parquet",
        columns=["i", "j", "t", "phat"],
    ).rename(columns={"phat": "mnl"})

    lightgbm = pd.read_parquet(
        f"{path_results}/LightGBM_Cross_ByJ.parquet",
        columns=["i", "j", "t", "phat"],
    ).rename(columns={"phat": "lightgbm"})

    dnn = pd.read_parquet(
        f"{path_results}/dnn_{model}_{args.e:08d}.parquet",
        columns=["i", "j", "t", "phat"],
    ).rename(columns={"phat": "dnn"})

    # overview
    df = (
        prediction_index.merge(dnn, on=["i", "t", "j"])
        .merge(freq_product, on=["i", "t", "j"])
        .merge(logit, on=["i", "t", "j"])
        .merge(lightgbm, on=["i", "t", "j"])
        .merge(mxl, on=["i", "t", "j"])
        .merge(mnl, on=["i", "t", "j"])
        .merge(actions, on=["i", "j", "t"], how="left")
    )
    logger.info(f"df.shape[0] = {df.shape[0]:,}")
    df["it_has_coupon"] = (df.groupby(["i", "t"]).discount.transform(sum) > 0).astype(
        int
    )
    df["ijt_has_coupon"] = (df.discount > 0).astype(int)

    # benchmarking
    _scores_list = Parallel(n_jobs=30)(
        delayed(score_models_rx)(rx, model_map, df) for rx in range(R)
    )
    scores = pd.concat(_scores_list).groupby(["model"]).agg(["mean", "std", "count"])
    scores.columns = ["_".join(x) for x in scores.columns]
    scores["log-loss_se"] = scores["log-loss_std"] / np.sqrt(scores["log-loss_count"])
    scores["auc_se"] = scores["auc_std"] / np.sqrt(scores["auc_count"])
    print(scores[["log-loss_mean", "log-loss_se"]])

    scores_format = scores.reset_index()[["model"]]
    scores_format["loss"] = scores[["log-loss_mean"]].round(5).values
    scores_format["delta"] = [
        f"{1 - x/scores_format[scores_format['model']=='mnl'].loss.values[0]:.1%}"
        for x in scores_format["loss"]
    ]

    # save
    df.to_parquet(f"{path_results}/predictions_overview.parquet")
    scores.to_csv(f"{path_results}/benchmarking.csv")

    scores_format.to_html(f"{path_results}/paper/table_6.html")
    scores_format.to_csv(f"{path_results}/paper/table_6.csv")
