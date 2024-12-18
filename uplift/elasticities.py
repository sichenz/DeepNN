# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import functools
import numpy as np
import pandas as pd
from loguru import logger

import modules.args
import modules.lib
import uplift.lib


def main(x, path_data, **kwargs):

    logger.info("Elasticity evaluation")

    # load config
    config = modules.lib.read_yaml(x)
    config["path_data"] = path_data
    logger.info(f"path_data={path_data}")
    config_prob = config["coupons"]
    dir_results = config_prob["dir_results"]
    path_output = f"{path_data}/{dir_results}"
    path_result = f"{path_output}/results"
    os.makedirs(path_result, exist_ok=True)
    discount = config_prob["discount"]
    I = config_prob["I"]
    models = config_prob["models"]
    for i, s in enumerate(models):
        logger.info(f"model {i} = {s}")
    all_suffixes = [m for m in models if m not in ["random", "uniform"]]
    logger.info(f"discount={discount}")
    logger.info(f"I={I}")

    # check state
    file_in = f"{path_data}/prob_uplift/combined_data.parquet"
    file_result = f"{path_data}/prob_uplift/results/elasticities_within.csv"
    if modules.lib.check_state(file_in, 1, file_result, path_data):
        return 0
    modules.lib.touch(file_result)

    # load data
    data_total = pd.read_parquet(f"{path_output}/combined_data.parquet")

    # update suffixes
    removed_suffixes = [
        s for s in all_suffixes if f"prob_{s}" not in data_total.columns
    ]
    if len(removed_suffixes) > 0:
        for s in removed_suffixes:
            logger.info(f"removed suffix `{s}`")
    all_suffixes = [s for s in all_suffixes if f"prob_{s}" in data_total.columns]
    for i, s in enumerate(all_suffixes):
        logger.info(f"suffix {i} = {s}")

    # compute probability uplift
    for s in all_suffixes:
        data_total.eval(f"prob_uplift_{s} = prob_{s} - prob_no_disc_{s}", inplace=True)

    # elasticity data
    data_el = data_total.groupby(["j", "coupon_j"]).agg(np.mean).reset_index()
    for s in all_suffixes:
        data_el[f"el_{s}"] = (
            data_el[f"prob_uplift_{s}"] / data_el[f"prob_no_disc_{s}"] / discount
        )
    all_el_variables = [f"el_{s}" for s in all_suffixes]
    data_el_save = data_el[
        ["j", "coupon_j", "category_group"] + all_el_variables
    ].reset_index(drop=True)
    data_el_save.to_parquet(f"{path_data}/prob_uplift/elasticities.parquet")

    # ELASTICITY
    elasticities_own = uplift.lib.single_overview_stats(
        x=data_el[data_el.category_group == 1],
        variables=all_el_variables,
        label="Own-Elasticities",
    )

    elasticities_within = uplift.lib.single_overview_stats(
        x=data_el[data_el.category_group == 2],
        variables=all_el_variables,
        label="Within-Elasticities",
    )

    elasticities_cross = uplift.lib.single_overview_stats(
        x=data_el[data_el.category_group == 3],
        variables=all_el_variables,
        label="Cross-Elasticities",
    )

    elasticities_cross_0 = uplift.lib.single_overview_stats(
        x=data_el[(data_el.category_group == 3) & (data_el.el_true == 0)],
        variables=all_el_variables,
        label="Cross-Elasticities (ZERO)",
    )

    elasticities_cross_pos = uplift.lib.single_overview_stats(
        x=data_el[(data_el.category_group == 3) & (data_el.el_true > 0)],
        variables=all_el_variables,
        label="Cross-Elasticities (POS)",
    )

    elasticities_cross_neg = uplift.lib.single_overview_stats(
        x=data_el[(data_el.category_group == 3) & (data_el.el_true < 0)],
        variables=all_el_variables,
        label="Cross-Elasticities (NEG)",
    )

    # save
    save_result_paths = functools.partial(
        uplift.lib.save_result,
        p=path_result,
    )
    save_result_paths(elasticities_own, file="elasticities_own")
    save_result_paths(elasticities_within, file="elasticities_within")
    save_result_paths(elasticities_cross, file="elasticities_cross")
    save_result_paths(elasticities_cross_0, file="elasticities_cross_0")
    save_result_paths(elasticities_cross_pos, file="elasticities_cross_pos")
    save_result_paths(elasticities_cross_neg, file="elasticities_cross_neg")

    logger.info("□")


if __name__ == "__main__":

    args = modules.args.global_args("Elasticity evaluation")
    all_path_data = modules.lib.get_data_paths(args)

    for name, params in all_path_data.items():
        main(x=args.c, **params)
