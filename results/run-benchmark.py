# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import tqdm
import numpy as np
import pandas as pd
from loguru import logger

from joblib import Parallel, delayed

import modules.args
import modules.lib
import modules.scores


def main(x, path_data, seed_offset, write_html=True, **kwargs):

    # load config
    config = modules.lib.read_yaml(x)
    config["path_data"] = path_data
    config["global_seed"] += seed_offset
    model_benchmarking_files = config["model-benchmarking"]["files"]
    logger.info(f"path_data={path_data}")
    logger.info(f"seed={seed_offset}")

    # load data
    baskets = pd.read_parquet(f"{path_data}/baskets.parquet")
    baskets["quantity"] = 1.0
    file_actions = f"{path_data}/action.parquet"
    if os.path.isfile(file_actions):
        actions = pd.read_parquet(file_actions)
    else:
        actions = pd.DataFrame({"i": [0], "j": 0, "t": 88, "discount": 0})
    data_j = pd.read_csv(f"{path_data}/data_j.csv", usecols=["j", "c"])
    J = data_j.shape[0]
    prod2id = {i: i for i in range(J)}

    # merge (and name) predictions
    prediction = pd.read_parquet(
        f"{path_data}/prediction_master_2.parquet"
    ).sort_values(["i", "j", "t"])
    for k, v in model_benchmarking_files.items():
        file_k = f"{path_data}/{v}"
        if not os.path.isfile(file_k):
            logger.warning(f"{file_k} does not exists (skipped)")
            continue
        if os.stat(file_k).st_size == 0:
            logger.warning(f"{file_k} is empty (skipped)")
            continue
        prediction_k = pd.read_parquet(
            f"{path_data}/{v}", columns=["i", "j", "t", "phat"]
        ).sort_values(["i", "j", "t"])
        prediction_k = prediction_k[prediction_k["i"].isin(prediction["i"].unique())]
        assert np.all(
            prediction[["i", "j", "t"]].values == prediction_k[["i", "j", "t"]].values
        )
        prediction[k] = prediction_k["phat"].values

    # merge actions
    logger.info("merge coupons")
    prediction = prediction.merge(
        actions[["i", "j", "t", "discount"]], on=["i", "j", "t"], how="left"
    )
    prediction["discount"].fillna(0, inplace=True)
    assert not np.any(prediction.isnull())

    # build indicators
    logger.info("build data subset indicators")
    # coupon for product
    prediction["product_discount"] = (prediction["discount"] > 0).astype(int)
    assert prediction[prediction["product_discount"] == 0]["discount"].sum() == 0
    # coupon in category
    prediction["category_discount"] = (
        prediction.groupby(["i", "c", "t"])["discount"].transform("sum") > 0
    ).astype(int)
    assert np.all(
        prediction[prediction["product_discount"] == 1]["category_discount"] == 1
    )
    # WITHIN
    prediction["within"] = (
        prediction["category_discount"] - prediction["product_discount"]
    )
    assert np.max(prediction.eval("product_discount + within")) == 1
    assert (prediction["within"].min() == 0) and (prediction["within"].max() == 1)
    # cross-category coupon (but not product or category coupon)
    prediction["cross_category_discount"] = (
        prediction["gamma_ick_cp_total"] > 0
    ).astype(int)
    # no discount (not even cross-effect)
    prediction["no_discount"] = (
        (prediction["category_discount"] + prediction["gamma_ick_cp_total"]) == 0
    ).astype(int)
    assert prediction[prediction["no_discount"] == 1]["discount"].sum() == 0
    assert prediction[prediction["no_discount"] == 1]["gamma_ick_cp_total"].sum() == 0
    # tmp = prediction[['i','c','j','t','discount','gamma_ick_cp_total']]
    # tmp[(tmp['i']==0) & (tmp['j']==0)]
    # time series with discount := user-category with discount (cross-effect is not enough)
    prediction["time_series_discount"] = (
        prediction.groupby(["i", "c"])["discount"].transform("sum") > 0
    ).astype(int)
    # time series without discount := not even a cross-effect
    prediction["time_series_no_discount"] = (
        (
            prediction.groupby(["i", "c"])["discount"].transform(sum)
            + prediction.groupby(["i", "c"])["gamma_ick_cp_total"].transform(sum)
        )
        == 0
    ).astype(int)
    assert np.all(
        prediction[prediction["time_series_no_discount"] == 1]
        .groupby(["i", "j"])
        .j.count()
        == 10
    )
    assert prediction[prediction["time_series_no_discount"] == 1]["discount"].sum() == 0
    assert (
        prediction[prediction["time_series_no_discount"] == 1][
            "gamma_ick_cp_total"
        ].sum()
        == 0
    )

    # data subsets
    x_pd = prediction[prediction["product_discount"] == 1]
    x_cd = prediction[prediction["within"] == 1]
    x_ccd = prediction[prediction["cross_category_discount"] == 1]
    x_tsd = prediction[prediction["time_series_discount"] == 1]
    x_tsnd = prediction[prediction["time_series_no_discount"] == 1]

    # metric computation (incl. SE)
    logger.info("compute metrics")
    models = ["p"] + [x for x in model_benchmarking_files.keys() if x in prediction]
    _benchmarking = Parallel(n_jobs=len(models))(
        delayed(modules.scores.get_scores)(
            prediction, x_pd, x_cd, x_ccd, x_tsd, x_tsnd, m
        )
        for m in models
    )
    benchmarking = pd.concat(_benchmarking)
    logger.info("metrics computation done")

    # save
    benchmarking.to_csv(f"{path_data}/benchmarking.csv", index=False)


if __name__ == "__main__":

    args = modules.args.global_args("Model evaluation.")
    all_path_data = modules.lib.get_data_paths(args)

    for name, params in all_path_data.items():
        main(x=args.c, **params)
