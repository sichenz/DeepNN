# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import numpy as np
import pandas as pd
from loguru import logger

import modules.args
import modules.lib
import uplift.lib


def main(x, path_data, **kwargs):

    logger.info("Combining probabilities for Probability Uplift")

    # load config
    config = modules.lib.read_yaml(x)
    config["path_data"] = path_data
    logger.info(f"path_data={path_data}")

    # check state
    # file_in = f"{path_data}/prob_uplift/total_prob_mxl_inv.parquet"
    file_in = f"{path_data}/prob_uplift/total_prob_true.parquet"
    path_output = f"{path_data}/prob_uplift"
    file_result = f"{path_output}/combined_data.parquet"
    if modules.lib.check_state(file_in, 1, file_result, path_data):
        return 0
    modules.lib.touch(file_result)

    # load data
    all_suffixes = [
        f"_{x}"
        for x in config["coupons"]["models"]
        if x not in ["true", "random", "uniform"]
        and os.path.isfile(f"{path_data}/prob_uplift/total_prob_{x}.parquet")
    ]
    for i, s in enumerate(all_suffixes):
        logger.info(f"suffix {i} = {s}")

    # build total probability table
    data_total = uplift.lib.load_all_probs(path_output, all_suffixes)
    # assert data_total.shape[0] == (config["coupons"]["I"] * config["coupons"]["J"] ** 2)

    # save
    os.makedirs(path_output, exist_ok=True)
    data_total.to_parquet(file_result)


if __name__ == "__main__":

    args = modules.args.global_args("Build probability base table.")
    all_path_data = modules.lib.get_data_paths(args)

    for name, params in all_path_data.items():
        main(x=args.c, **params)
