# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import torch
import sklearn.metrics
import numpy as np
import pandas as pd
from loguru import logger

import modules.args
import modules.lib
import modules.predictor


def main(x, path_data, model, suffix, epoch, rerun=False, **kwargs):

    logger.info("Predict `dnn`")

    logger.info(f"path_data={path_data}")
    logger.info(f"model={model}")
    logger.info(f"suffix={suffix}")
    logger.info(f"epoch={epoch}")
    config = modules.lib.read_yaml(x)
    use_gpu = config["use_gpu"]
    logger.info(f"use_gpu={use_gpu}")

    # check state
    file_sd = f"{path_data}/{model}{suffix}/results/state_dict_{epoch:08d}.pt"
    file_result = (
        f"{path_data}/{model}{suffix}/predicted_probabilities_{epoch:08d}.parquet"
    )
    if not rerun:
        if modules.lib.check_state(file_sd, 1e-8, file_result, path_data):
            return 0
    else:
        logger.info("Forcing making predictions (possibly overwriting old results)")
    modules.lib.touch(file_result)

    # load data
    test_set_config = config["test_set"]
    prediction_master = pd.read_parquet(f"{path_data}/prediction_master_2.parquet")
    baskets = pd.read_parquet(f"{path_data}/baskets.parquet")
    baskets["quantity"] = 1.0
    file_actions = f"{path_data}/action.parquet"
    if os.path.isfile(file_actions):
        actions = pd.read_parquet(file_actions)
    else:
        actions = pd.DataFrame(
            {"i": [prediction_master.i[0]], "j": 0, "t": 88, "discount": 0}
        )
    data_j = pd.read_csv(f"{path_data}/data_j.csv")
    J = data_j.shape[0]
    prod2id = {i: i for i in range(J)}

    # main
    torch.manual_seed(501)
    np.random.seed(501)

    # prune data
    logger.info("subset data")
    users = prediction_master.i.unique()
    baskets_users = baskets[baskets["i"].isin(users)].reset_index(drop=True)
    actions_users = actions[actions["i"].isin(users)].reset_index(drop=True)

    # prediction
    logger.info("make prediction")
    res = modules.predictor._predictor_core(
        path=path_data,
        model=model,
        suffix=suffix,
        epoch=epoch,
        prediction_master=prediction_master,
        baskets=baskets_users,
        actions=actions_users,
        J=J,
        time_first=test_set_config["t_start"],
        time_last=test_set_config["t_end"],
        use_gpu=use_gpu,
    )

    # log-loss
    logger.info(f"log-loss = {sklearn.metrics.log_loss(res['y'], res['phat']):.6f}")

    # save
    res.to_parquet(file_result)

    logger.info("□")


if __name__ == "__main__":

    args = modules.args.global_args("DNN predict.")
    all_path_data = modules.lib.get_data_paths(args, rerun=True)

    for name, params in all_path_data.items():
        if "epoch" in params:
            logger.info("reading epoch from params")
            args.e = params["epoch"]
            del params["epoch"]
        main(x=args.c, rerun=args.r, epoch=args.e, **params)
