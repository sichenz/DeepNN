# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import gc
import re
import numpy as np
import pandas as pd
from loguru import logger


def load_computed_probs(filename, suffix):

    total_prob = pd.read_parquet(filename)

    idx_discount = total_prob.discount == "discount"
    result = total_prob.loc[idx_discount][["i", "j", "coupon_j", "price_paid", "prob"]]
    result_no_disc = total_prob.loc[~idx_discount][
        ["i", "j", "coupon_j", "price_paid", "prob"]
    ]
    result_no_disc.columns = ["i", "j", "coupon_j", "price_paid", "prob_no_disc"]

    result.eval("rev = prob*price_paid", inplace=True)
    result_no_disc.eval("rev_no_disc = prob_no_disc*price_paid", inplace=True)

    result = result.merge(
        result_no_disc[["i", "j", "prob_no_disc", "rev_no_disc"]], on=["i", "j"]
    )

    result = result[
        [
            "i",
            "j",
            "coupon_j",
            "prob",
            "prob_no_disc",
            "rev",
            "rev_no_disc",
        ]
    ]

    result.columns = [
        "i",
        "j",
        "coupon_j",
        "prob" + suffix,
        "prob_no_disc" + suffix,
        "rev" + suffix,
        "rev_no_disc" + suffix,
    ]

    return result


def load_all_probs(path_output, all_suffixes, file_prefix="total_prob", ref="_true"):

    logger.info(f"reference={ref}")
    logger.info(f"load {ref[1:]}")
    data_total = load_computed_probs(f"{path_output}/{file_prefix}{ref}.parquet", ref)

    for suffix in all_suffixes:
        # suffix = all_suffixes[0]
        logger.info(f"load {suffix[1:]}")  # [1:] removes `_`
        data_iter = load_computed_probs(
            f"{path_output}/{file_prefix}{suffix}.parquet", suffix
        )

        assert np.all(data_total.index.values == data_iter.index.values)
        assert np.all(
            data_total[["i", "j", "coupon_j"]].values
            == data_iter[["i", "j", "coupon_j"]].values
        )
        data_iter.drop(columns=["i", "j", "coupon_j"], inplace=True)
        data_total = pd.concat([data_total, data_iter], axis=1)
        del data_iter
        gc.collect()

        data_total["c"] = data_total["j"] // 10
        data_total["coupon_c"] = data_total["coupon_j"] // 10
        data_total.eval("own_product = (j==coupon_j)*1", inplace=True)
        data_total.eval("within_category = (c==coupon_c)*1 - own_product", inplace=True)
        data_total.eval("cross_category = (c!=coupon_c)*1", inplace=True)
        assert np.all(
            data_total.eval("own_product + within_category + cross_category") == 1
        )

        data_total["category_group"] = (
            1 * data_total.own_product
            + 2 * data_total.within_category
            + 3 * data_total.cross_category
        )

    return data_total


def overview_stats(x, variables, label, metric=None):
    idx_own = x.category_group == 1
    idx_within = x.category_group == 2
    idx_cross = x.category_group == 3
    _list = []
    x_sub = x[variables].reset_index(drop=True)
    ref_variable = [v for v in variables if re.match(".*_true$", v)]
    assert len(ref_variable) == 1
    for var in variables:
        if metric != np.mean:
            x_sub[var] = x_sub[var].values - x_sub[ref_variable[0]].values
        _list.append(
            pd.DataFrame(
                {
                    "var": [var],
                    "All": metric(x_sub[var]),
                    "Own": metric(x_sub[idx_own][var]),
                    "Within": metric(x_sub[idx_within][var]),
                    "Cross": metric(x_sub[idx_cross][var]),
                }
            )
        )
    result = pd.concat(_list).reset_index(drop=True)
    print(label)
    print(result.round(4))
    return result


def single_overview_stats(x, variables, label):
    ref_variable = [v for v in variables if re.match(".*_true$", v)]
    assert len(ref_variable) == 1
    _list = []
    for var in variables:
        _list.append(
            pd.DataFrame(
                {
                    "var": [var],
                    "Mean": np.mean(x[var]),
                    "MAE": mae(x[var] - x[ref_variable[0]]),
                    "RMSE": rmse(x[var] - x[ref_variable[0]]),
                }
            )
        )
    result = pd.concat(_list).reset_index(drop=True)
    print(label)
    print(result.round(4))
    return result


def mae(x):
    return np.mean(np.abs(x))


def rmse(x):
    return np.sqrt(np.mean(x ** 2))


def save_result(x, p, file):
    x.to_csv(f"{p}/{file}.csv")
