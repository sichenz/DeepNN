# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import numpy as np
import pandas as pd
from loguru import logger

import modules.args
import modules.lib


def purchase_frequency(by, norm, prediction_master, baskets, fillna=False):
    _tmp = []
    for t_i in range(prediction_master.t.min(), prediction_master.t.max() + 1, 1):
        baskets_t = baskets[baskets["t"] < t_i]
        if by is None:
            pf_t = pd.DataFrame({"phat": [baskets_t.shape[0] / (t_i * norm)]})
        else:
            pf_t = (
                (baskets_t.groupby(by)[["t"]].count() / (t_i * norm))
                .rename(columns={"t": "phat"})
                .reset_index()
            )
        pf_t["t"] = t_i
        _tmp.append(pf_t)
    out = prediction_master.merge(
        pd.concat(_tmp), on=["t"] if by is None else ["t"] + by, how="left"
    )
    if fillna:
        out.phat.fillna(0, inplace=True)
    else:
        assert np.all(out.phat.notnull())
    return out


def main(x, path_data, seed_offset, **kwargs):

    logger.info("Baseline `frequency`")

    # load config
    config = modules.lib.read_yaml(x)
    config["path_data"] = path_data
    logger.info(f"path_data={path_data}")
    os.makedirs(f"{path_data}/baselines", exist_ok=True)

    # check state
    file_master = f"{path_data}/prediction_master_2.parquet"
    file_result = f"{path_data}/baselines/ProductPF.parquet"
    if modules.lib.check_state(file_master, 1, file_result, path_data):
        return 0
    modules.lib.touch(file_result)

    # load data
    prediction_master = pd.read_parquet(file_master)
    baskets = pd.read_parquet(f"{path_data}/baskets.parquet")
    J = prediction_master.j.max() + 1
    I = 5_000
    baskets = baskets[baskets.i < I]

    # product purchase frequency
    pf_product = purchase_frequency(
        by=["j"],
        norm=I,
        prediction_master=prediction_master,
        baskets=baskets,
        fillna=True,
    )

    # save
    pf_product.to_parquet(file_result)

    logger.info("□")


if __name__ == "__main__":

    args = modules.args.global_args("Baseline `frequency`")
    all_path_data = modules.lib.get_data_paths(args)

    for name, params in all_path_data.items():
        if modules.lib.check_bootstrap_iter(params["seed_offset"], args.y, args.z):
            continue
        main(x=args.c, **params)
