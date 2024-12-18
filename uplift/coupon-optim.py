# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import argparse
import functools
import numpy as np
import pandas as pd
from loguru import logger

import modules.args
import modules.lib


def pick_best(x, vo, vm, n_check, by=["i"], eps=1e-12):
    x_sub = x[np.unique([vo, vm, "i"])].reset_index(drop=True)
    x_sub[vo] += np.random.uniform(-eps, eps, x.shape[0])
    optim_result = x_sub.sort_values(vo, ascending=False).groupby("i").head(1)
    assert optim_result.shape[0] == n_check
    return optim_result[vm].mean()


def pick_uniform(x, vo, vm, n_check, eps=1e-6):
    x_sub = x[np.unique([vo, vm, "coupon_j", "i"])].reset_index(drop=True)
    x_sub["global_score"] = x_sub.groupby("coupon_j")[vo].transform(sum)
    noise = x_sub[["coupon_j"]].drop_duplicates()
    noise["eps"] = np.random.uniform(-eps, eps, noise.shape[0])
    x_sub = x_sub.merge(noise, on="coupon_j", how="left")
    assert np.all(x_sub.eps.notnull())
    x_sub["global_score"] = x_sub["global_score"] + x_sub["eps"]
    assert np.all(x_sub.groupby("coupon_j").global_score.nunique() == 1)
    assert np.all(x_sub.groupby("global_score").i.count() == n_check)
    optim_result = (
        x_sub.sort_values("global_score", ascending=False).groupby("i").head(1)
    )
    assert optim_result.shape[0] == n_check
    assert optim_result.coupon_j.nunique() == 1
    return optim_result[vm].mean()


def pick_random(x, vm):
    return x[vm].mean()


def run_optim(
    data_total,
    data_category_g1,
    data_category_g2,
    data_category_g3,
    I,
    do_random,
    do_uniform,
    var_ref,
    var_iter_all,
    var_ref_uniform,
    var_ref_uniform2,
    label,
    p,
    file,
):
    print(label)
    optimization_result = []
    for var_iter in var_iter_all:
        optimization_result.append(
            [
                var_iter,
                pick_best(data_total, var_iter, var_ref, I),
                pick_best(data_category_g1, var_iter, var_ref, I),
                pick_best(data_category_g2, var_iter, var_ref, I),
                pick_best(data_category_g3, var_iter, var_ref, I),
            ]
        )
    if do_random:
        optimization_result.append(
            [
                "random",
                pick_random(data_total, var_ref),
                pick_random(data_category_g1, var_ref),
                pick_random(data_category_g2, var_ref),
                pick_random(data_category_g3, var_ref),
            ]
        )
    if do_uniform:
        optimization_result.append(
            [
                "uniform-true",
                pick_uniform(data_total, var_ref_uniform2, var_ref, I),
                pick_uniform(data_category_g1, var_ref_uniform2, var_ref, I),
                pick_uniform(data_category_g2, var_ref_uniform2, var_ref, I),
                pick_uniform(data_category_g3, var_ref_uniform2, var_ref, I),
            ]
        )

    optimization_result = pd.DataFrame(
        optimization_result,
        columns=["var", "Total", "Own", "Within", "Cross"],
    ).round(decimals=4)
    print(optimization_result)
    optimization_result.to_csv(f"{p}/{file}.csv")
    return optimization_result


def main(x, path_data, dropbox=False, **kwargs):

    logger.info("Coupon optimization")

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
    do_random = "random" in models
    do_uniform = "uniform" in models
    logger.info(f"discount={discount}")
    logger.info(f"I={I}")

    # check state
    file_in = f"{path_data}/prob_uplift/combined_data.parquet"
    file_result = f"{path_data}/prob_uplift/results/optim_revenue_uplift.csv"
    if modules.lib.check_state(file_in, 1, file_result, path_data):
        return 0
    modules.lib.touch(file_result)

    # load data
    data_probs = pd.read_parquet(f"{path_output}/combined_data.parquet")
    data_j = pd.read_csv(f"{path_data}/data_j.csv")
    # J = 250
    J = data_j.shape[0]

    # update suffixes
    removed_suffixes = [
        s for s in all_suffixes if f"prob_{s}" not in data_probs.columns
    ]
    if len(removed_suffixes) > 0:
        for s in removed_suffixes:
            logger.info(f"removed suffix `{s}`")
    all_suffixes = [s for s in all_suffixes if f"prob_{s}" in data_probs.columns]
    for i, s in enumerate(all_suffixes):
        logger.info(f"suffix {i} = {s}")

    # aggregated data versions
    data_category = (
        data_probs.groupby(["i", "coupon_j", "category_group"]).agg(sum).reset_index()
    )
    # compute uplift variables
    for s in all_suffixes:
        data_category.eval(
            f"prob_uplift_{s} = prob_{s} - prob_no_disc_{s}", inplace=True
        )
        data_category.eval(f"rev_uplift_{s} = rev_{s} - rev_no_disc_{s}", inplace=True)
    assert data_category.shape[0] == (I * J * 3)
    assert data_category.shape[0] == (I * J * 3)
    data_category_g1 = data_category[data_category["category_group"] == 1]
    data_category_g2 = data_category[data_category["category_group"] == 2]
    data_category_g3 = data_category[data_category["category_group"] == 3]
    assert data_category_g1.shape[0] == (I * J)
    assert data_category_g2.shape[0] == (I * J)
    assert data_category_g3.shape[0] == (I * J)
    del data_probs

    data_total = data_category.groupby(["i", "coupon_j"]).agg(sum).reset_index()
    assert data_total.shape[0] == (I * J)

    run_optim_data = functools.partial(
        run_optim,
        data_total=data_total,
        data_category_g1=data_category_g1,
        data_category_g2=data_category_g2,
        data_category_g3=data_category_g3,
        I=I,
        do_random=do_random,
        do_uniform=do_uniform,
        p=path_result,
    )

    # Optimization
    res_probability = run_optim_data(
        var_ref="prob_true",
        var_iter_all=[f"prob_{s}" for s in all_suffixes] + ["prob_no_disc_true"],
        var_ref_uniform="prob_logit_cross_by_j",
        var_ref_uniform2="prob_true",
        label="Optimize Probability",
        file="optim_probability",
    )

    res_probability_uplift = run_optim_data(
        var_ref="prob_uplift_true",
        var_iter_all=[f"prob_uplift_{s}" for s in all_suffixes],
        var_ref_uniform="prob_uplift_logit_cross_by_j",
        var_ref_uniform2="prob_uplift_true",
        label="Optimize Probability Uplift",
        file="optim_probability_uplift",
    )

    res_revenue = run_optim_data(
        var_ref="rev_true",
        var_iter_all=[f"rev_{s}" for s in all_suffixes] + ["rev_no_disc_true"],
        var_ref_uniform="rev_logit_cross_by_j",
        var_ref_uniform2="rev_true",
        label="Optimize Revenue",
        file="optim_revenue",
    )

    res_prob_uplift = run_optim_data(
        var_ref="rev_uplift_true",
        var_iter_all=[f"rev_uplift_{s}" for s in all_suffixes],
        var_ref_uniform="rev_uplift_logit_cross_by_j",
        var_ref_uniform2="rev_uplift_true",
        label="Optimize Revenue Uplift",
        file="optim_revenue_uplift",
    )

    logger.info("□")


if __name__ == "__main__":

    args = modules.args.global_args("Coupon optimization")
    all_path_data = modules.lib.get_data_paths(args)

    for name, params in all_path_data.items():
        main(x=args.c, dropbox=args.s, **params)
