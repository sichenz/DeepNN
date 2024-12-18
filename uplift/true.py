# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import scipy
import copy
import pickle
import numpy as np
import pandas as pd
from loguru import logger

import modules.args
import modules.lib
import modules.simulation


def main(x, path_data, **kwargs):

    logger.info("Discount simulation `true`")

    # load config
    config = modules.lib.read_yaml(x)
    config["path_data"] = path_data
    path_output = f"{path_data}/prob_uplift"
    config_prob = config["coupons"]
    I = config_prob["I"]
    discount = config_prob["discount"]
    logger.info(f"path_data={path_data}")
    logger.info(f"I={I}")
    logger.info(f"discount={discount}")

    # check state
    file_gym0 = f"{path_data}/gym0_light.pickle.gz"
    file_result = f"{path_output}/total_prob_true.parquet"
    if modules.lib.check_state(file_gym0, 10, file_result, path_data):
        return 0
    os.makedirs(path_output, exist_ok=True)
    modules.lib.touch(file_result)

    # load data
    baskets = pd.read_parquet(f"{path_data}/baskets.parquet")
    baskets["y"] = 1

    data_c = pd.read_csv(f"{path_data}/data_c.csv")
    data_j = pd.read_csv(f"{path_data}/data_j.csv")
    J = data_j.shape[0]
    logger.info(f"J={J}")

    # load GYM
    logger.info("load gym")
    gym = modules.lib.load_gzip_pickle(file_gym0)
    logger.info("recreate gym state")
    gym_100 = gym.recreate_gym_state(gym, baskets, 100)
    assert gym.week == 100
    gym.I = I
    gym.data_ic = gym.data_ic.loc[range(I)]
    gym.data_ij = gym.data_ij.loc[range(I)]
    gym.data_ij_idx = gym.data_ij.reset_index()[["i", "j"]]
    gym._precomputed_gumbel = np.random.choice(
        gym._precomputed_gumbel, gym.data_ij.shape[0]
    )
    gym.gamma_ick_cp = gym.gamma_ick_cp[np.arange(gym.I * gym.C), :]
    assert gym.data_ic.reset_index().i.nunique() == I
    assert gym.data_ij.reset_index().i.nunique() == I

    # extract probs
    logger.info("simulate (true) uplift")
    prob = []

    # with discount
    for jn in range(J):
        action = pd.DataFrame({"i": list(range(I)), "j": jn, "discount": discount})

        gym_copy = copy.deepcopy(gym)
        _ = gym_copy.generate(T=1, action=action, seed_offset=0)

        data_ic = gym_copy.data_ic.reset_index()
        data_ic["m2_p_ict"] = scipy.stats.norm.cdf(data_ic["m2_v_ic"])
        data_ij = gym_copy.data_ij.reset_index()

        er_ijt = pd.merge(
            data_ij[["i", "c", "j", "m3_p_ijt", "d_ijt", "price_paid"]],
            data_ic[["i", "c", "m2_p_ict"]],
            on=["i", "c"],
            how="left",
        )

        p_ijt = er_ijt.eval("prob = m2_p_ict * m3_p_ijt")
        p_ijt["coupon_j"] = jn
        p_ijt["discount"] = "discount"

        prob.append(p_ijt[["i", "j", "coupon_j", "discount", "price_paid", "prob"]])

    # without discount
    action = pd.DataFrame({"i": list(range(I)), "j": 0, "discount": 0})

    gym_copy = copy.deepcopy(gym)
    _ = gym_copy.generate(T=1, action=action, seed_offset=0)
    data_ic = gym_copy.data_ic.reset_index()
    data_ic["m2_p_ict"] = scipy.stats.norm.cdf(data_ic["m2_v_ic"])
    data_ij = gym_copy.data_ij.reset_index()

    er_ijt = pd.merge(
        data_ij[["i", "c", "j", "m3_p_ijt", "d_ijt", "price_paid"]],
        data_ic[["i", "c", "m2_p_ict"]],
        on=["i", "c"],
        how="left",
    )

    p_ijt = er_ijt.eval("prob = m2_p_ict * m3_p_ijt")
    p_ijt["coupon_j"] = 0
    p_ijt["discount"] = "no discount"

    prob.append(p_ijt[["i", "j", "coupon_j", "discount", "price_paid", "prob"]])

    # save
    total_prob = pd.concat(prob)
    total_prob.to_parquet(file_result)

    logger.info("□")


if __name__ == "__main__":

    args = modules.args.global_args("Discount simulation `true`")
    all_path_data = modules.lib.get_data_paths(args)

    for name, params in all_path_data.items():
        if modules.lib.check_bootstrap_iter(params["seed_offset"], args.y, args.z):
            continue
        main(x=args.c, **params)
