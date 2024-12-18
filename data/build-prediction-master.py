# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import scipy.stats
import numpy as np
import pandas as pd
from loguru import logger

import modules.args
import modules.lib as lib
import modules.simulation


def extract_data_from_gym(x, baskets, set_is, target_week, W, action=None):

    logger.info("recreate gym state")
    x2 = x.recreate_gym_state(x=x, baskets=baskets, target_week=target_week)

    logger.info(f"rerun {W} weeks [{target_week}, {target_week+W-1}]")
    _tmp_baskets = []
    _tmp_data_ic = []
    _tmp_data_ij = []
    for t in range(W):
        baskets_t = x2.generate(T=1, action=action)
        _tmp_baskets.append(baskets_t[baskets_t["i"].isin(set_is)])
        data_ic_t = x2.data_ic[
            ["m2_u_ic", "m2_v_ic", "inv", "eps_ict", "gamma_ick_cp_total"]
        ].reset_index()
        data_ic_t["t"] = baskets_t["t"].values[0]
        data_ic_t["m2_p_ict"] = scipy.stats.norm.cdf(data_ic_t["m2_v_ic"])
        _tmp_data_ic.append(data_ic_t[data_ic_t["i"].isin(set_is)])
        data_ij_t = x2.data_ij[
            [
                "c",
                "m3_p_ijt",
                "m3_u_ijt",
                "m3_v_ijt",
                "eps_ijt",
                "d_ijt",
                "price_paid",
            ]
        ].reset_index()
        data_ij_t["t"] = baskets_t["t"].values[0]
        _tmp_data_ij.append(data_ij_t[data_ij_t["i"].isin(set_is)])
    return pd.concat(_tmp_baskets), pd.concat(_tmp_data_ic), pd.concat(_tmp_data_ij)


def main(x, path_data, seed_offset, **kwargs):

    # load config
    config = modules.lib.read_yaml(x)
    config["path_data"] = path_data
    logger.info(f"path_data={path_data}")
    test_config = config["test_set"]

    # check whether results exists
    file_gym0 = f"{path_data}/gym0_light.pickle.gz"
    file_result = f"{path_data}/prediction_master_2.parquet"
    if modules.lib.check_state(file_gym0, 10, file_result, path_data):
        return 0
    modules.lib.touch(file_result)

    # load data
    gym0 = modules.lib.load_gzip_pickle(file_gym0)
    baskets = pd.read_parquet(f"{path_data}/baskets.parquet")
    baskets["y"] = 1
    file_action = f"{path_data}/action.parquet"
    if os.path.isfile(file_action):
        action = pd.read_parquet(file_action)
    else:
        action = None

    # extract probabilities from gym0 (data_ij and data_ic)
    # extract baskets to test whether running the gym0 a second time yields the same results
    baskets_tests, data_ic, data_ij = extract_data_from_gym(
        x=gym0,
        baskets=baskets,
        action=action,
        set_is=range(test_config["I"]),
        target_week=test_config["t_start"],
        W=test_config["t_end"] - test_config["t_start"] + 1,
    )

    # test that baskets are identical across the two runs
    logger.info("test output")
    test_df_1 = baskets[
        (baskets.t >= test_config["t_start"]) & (baskets.i < test_config["I"])
    ].reset_index(drop=True)
    test_df_2 = baskets_tests.reset_index(drop=True)
    pd.testing.assert_frame_equal(
        test_df_1[["i", "j", "p_jc", "price_paid", "t"]],
        test_df_2[["i", "j", "p_jc", "price_paid", "t"]],
    )

    # build prediction master
    logger.info("build prediction master")
    prediction_master = (
        data_ij[["i", "c", "j", "t", "m3_p_ijt"]]
        .merge(
            data_ic[["i", "c", "t", "m2_p_ict", "gamma_ick_cp_total"]],
            on=["i", "c", "t"],
            how="left",
        )
        .merge(baskets[["i", "t", "j", "y"]], on=["i", "t", "j"], how="left")
    )
    prediction_master.eval("p = m2_p_ict * m3_p_ijt", inplace=True)
    prediction_master["y"] = prediction_master["y"].fillna(0).astype(int)

    # Compare to old prediction master
    file_result_old = f"{path_data}/prediction_master.parquet"
    if os.path.isfile(file_result_old):
        logger.info("comparing new prediction master to old prediction master")
        prediction_master_old = pd.read_parquet(file_result_old)
        assert (
            np.abs(
                np.max(
                    prediction_master_old[["i", "c", "t", "p", "y"]].values
                    - prediction_master[["i", "c", "t", "p", "y"]].values
                )
            )
            < 1e-8
        )

    # save results
    prediction_master.to_parquet(file_result)

    logger.info("□")


if __name__ == "__main__":

    args = modules.args.global_args("Build prediction master.")
    all_path_data = modules.lib.get_data_paths(args)

    for name, params in all_path_data.items():
        if modules.lib.check_bootstrap_iter(params["seed_offset"], args.y, args.z):
            continue
        main(x=args.c, **params)
