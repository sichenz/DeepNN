# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import shutil
import copy
import numpy as np
import pandas as pd
from loguru import logger

import modules.args
import modules.lib
import modules.agents
import modules.simulation


def subset_coupons(x, f):
    if f == 1:
        return x
    elif f == 0:
        return None
    else:
        idx = np.random.choice(x.shape[0], int(f * x.shape[0]), replace=False)
        return x.iloc[idx].reset_index(drop=True)


def main(x, parameter_set, path_data, seed_offset, fraction_discounts=1, **kwargs):

    # CONFIG
    logger.info(f"parameter_set={parameter_set}")
    global_config = modules.lib.read_yaml(x)
    config = global_config["simulated_data"][parameter_set]
    path_data = os.path.expandvars(path_data)

    logger.info(f"path_data={path_data}")
    file_result = f"{path_data}/gym0_light.pickle"
    if os.path.isfile(f"{file_result}.gz"):
        logger.warning(f"`gym0_light.pickle.gz` exists, skipping {path_data}...")
        return 0
    os.makedirs(path_data)
    modules.lib.touch(file_result)
    shutil.copyfile(x, f"{path_data}/config.yaml")
    config["seed"] += seed_offset
    logger.info(f"seed={config['seed']}")

    if "I" in kwargs:
        I = kwargs.get("I")
        logger.info(f"I = {I:,}")
        config["I"] = I
    if "J" in kwargs:
        J = kwargs.get("J")
        logger.info(f"J = {J:,}")

    # some key variables
    variables_print = [
        "mu_ps",
        "sigma_ps",
        "cons_c_min",
        "cons_c_max",
        "gamma_c_inv_mu",
        "gamma_c_inv_sigma",
        "mu_gamma_p",
        "sigma_gamma_p",
        "prob_cp",
        "delta_cp",
        "own_price_method",
    ]
    for v in variables_print:
        logger.info(f"{v} = {config[v]}")

    # MASTER DATA
    data_c_raw = pd.read_csv(config["file_data_c"])
    data_c = data_c_raw[[x for x in ["c", "cons_c"] if x in data_c_raw]].reset_index(
        drop=True
    )
    C = data_c.c.nunique()
    data_j_raw = pd.read_csv(config["file_data_j"])[["j", "c"]]
    if "J" in config:
        J = config["J"]
        logger.info(f"Modify number of products ({data_j_raw.shape[0]} -> {J})")
        # for modifying J:
        # keep number of categories constant (to ensure comparable basket sizes)
        # only change the number of products by category
        J_max = data_j_raw.j.nunique() * 5
        if J > data_j_raw.j.nunique() * 5:
            raise Exception(
                f"J (= {J}) too large, maximum allowed value is J_max = {J_max}"
            )
        assert J % C == 0
        J_C = J // C
        data_j_0 = (
            pd.concat([data_j_raw for i in range(5)])
            .sort_values(["c"], ascending=[True])
            .reset_index(drop=True)
        )
        data_j_0["j_in_c"] = data_j_0.groupby("c").cumcount()
        data_j = data_j_0[data_j_0["j_in_c"] < J_C].reset_index(drop=True)
        data_j["j"] = range(J)
        data_j = data_j.reset_index(drop=True).copy()
    else:
        data_j = data_j_raw
    J = data_j.j.nunique()
    assert data_j.c.min() == 0
    assert data_j.c.max() == (data_j.c.nunique() - 1)
    assert data_j.j.min() == 0
    assert data_j.j.max() == (data_j.j.nunique() - 1)
    logger.info(f"J = {J}")
    logger.info(f"C = {C}")

    # COUPONS
    logger.info(f"create coupons (random agent)")

    # seed for coupons
    np.random.seed(config["seed"])

    # initialize agent
    random_agent = modules.agents.RandomAgentN(
        data_j=data_j,
        I=config["I"],
        discounts=config["discounts"],
        n_coupons=config["n_coupons"],
    )

    # build burnin actions
    burnin_action = []
    for t in range(1, config["burn_in"] + 1, 1):
        action_t = random_agent.feed(seed_offset=-t)
        action_t["t"] = -t
        burnin_action.append(action_t)
    burnin_action = pd.concat(burnin_action)

    # build actions
    action = []
    for t in range(config["T"]):
        action_t = random_agent.feed(seed_offset=t)
        action_t["t"] = t
        action.append(action_t)
    action = pd.concat(action)

    # reduce actions
    logger.info(f"fraction_discounts = {fraction_discounts}")
    burnin_action = subset_coupons(burnin_action, fraction_discounts)
    action = subset_coupons(action, fraction_discounts)

    # logging
    if burnin_action is None:
        logger.info("No burnin coupons")
    else:
        logger.info(f"n(burnin coupons) = {burnin_action.shape[0]:,}")
    if action is None:
        logger.info("No coupons")
    else:
        logger.info(f"n(coupons) = {action.shape[0]:,}")

    # INIT
    gym = modules.simulation.SupermarketGym(
        I=config["I"],
        data_c=data_c,
        data_j=data_j,
        burnin_action=burnin_action,
        w_i_cons_min=-1,
        w_i_cons_max=1,
        delta_W_cons=0.2,
        n_hidden=20,
        mu_ps=config["mu_ps"],
        sigma_ps=config["sigma_ps"],
        cons_c_min=config["cons_c_min"],
        cons_c_max=config["cons_c_max"],
        gamma_c_min=-0.9,
        gamma_c_max=0.4,
        mu_p=6,
        sigma_p=0.4,
        gamma_c_inv_mu=config["gamma_c_inv_mu"],
        gamma_c_inv_sigma=config["gamma_c_inv_sigma"],
        lambda_0_inv=0.4,
        delta_cons=1,
        scale_Gamma_c=1,
        mu_gamma_p=config["mu_gamma_p"],
        sigma_gamma_p=config["sigma_gamma_p"],
        cp_type=config["cp_type"],
        delta_cp=config["delta_cp"],
        sigma_cp=config["sigma_cp"],
        prob_cp=config["prob_cp"],
        mu_Beta_jc_min=0,
        mu_Beta_jc_max=2,
        scale_Beta_jc=4,
        burn_in=config["burn_in"],
        own_price_method=config["own_price_method"],
        seed=config["seed"],
    )
    logger.info("gym init done")

    if gym.save_data_burnin:
        baskets_burnin = gym.data_burnin.merge(
            gym.data_j[["c", "j"]], on="j", how="left"
        )

    gym0 = copy.deepcopy(gym)

    # GENERATE DATA
    baskets = gym.generate(T=config["T"], action=action)
    logger.info("gym.generate done")

    # SAVE
    logger.info("save data")
    baskets.to_parquet(f"{path_data}/baskets.parquet")
    # if gym.save_data_burnin:
    #    baskets_burnin.to_parquet(f"{path_data}/baskets_burnin.parquet")
    if action is not None:
        action.reset_index(drop=True).to_parquet(f"{path_data}/action.parquet")
    # if burnin_action is not None:
    #    burnin_action.reset_index(drop=True).to_parquet(
    #        f"{path_data}/burnin_action.parquet"
    #    )
    gym.data_c.to_csv(f"{path_data}/data_c.csv", index=False)
    gym.data_j.to_csv(f"{path_data}/data_j.csv", index=False)
    gym0.save(f"{path_data}/gym0_light.pickle")
    os.system(f"gzip {path_data}/gym0_light.pickle")

    # create data for week T+1
    if action is not None:
        action_TP1 = action[action["t"] == 0].reset_index(drop=True)
        action_TP1["t"] = gym.week
        action_TP1.to_parquet(f"{path_data}/actions_TP1.parquet")
    else:
        action_TP1 = None
    baskets_TP1 = gym.generate(T=1, action=action_TP1)
    baskets_TP1.to_parquet(f"{path_data}/baskets_TP1.parquet")
    logger.info("data saved")

    logger.info("□")


if __name__ == "__main__":

    args = modules.args.global_args("Build data set")
    all_path_data = modules.lib.get_data_paths(args, create_data=True)

    for name, params in all_path_data.items():
        if modules.lib.check_bootstrap_iter(params["seed_offset"], args.y, args.z):
            continue
        main(x=args.c, parameter_set=args.d, **params)
