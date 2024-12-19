# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import importlib
import torch
import numpy as np
import pandas as pd
from loguru import logger

import modules.args
import modules.lib
import modules.trainer
import modules.data_streamer_v1


def main(x, path_data, model, **kwargs):

    logger.info("Discount simulation `dnn`")

    # load config
    config = modules.lib.read_yaml(x)
    config["path_data"] = path_data
    path_output = f"{path_data}/prob_uplift"
    os.makedirs(path_output, exist_ok=True)
    config_prob = config["coupons"]
    I = config_prob["I"]
    discount = config_prob["discount"]
    epoch = config_prob["epoch"]
    t0 = config_prob["t0"]
    logger.info(f"path_data={path_data}")
    logger.info(f"I={I}")
    logger.info(f"t0={t0}")
    logger.info(f"discount={discount}")
    logger.info(f"model={model}")
    logger.info(f"epoch={epoch}")

    # check state
    file_sd = f"{path_data}/{model}/results/state_dict_{epoch:08d}.pt"
    file_result = f"{path_output}/total_prob_dnn_{model}.parquet"
    if modules.lib.check_state(file_sd, 1e-8, file_result, path_data):
        return 0
    modules.lib.touch(file_result)

    # load data
    file_actions = f"{path_data}/action.parquet"
    if os.path.isfile(file_actions):
        actions = pd.read_parquet(file_actions)
    else:
        logger.warning("no discounts, skipping")
        os.remove(file_result)
        return 0
    baskets = pd.read_parquet(f"{path_data}/baskets.parquet")
    baskets["quantity"] = 1.0

    data_j = pd.read_csv(f"{path_data}/data_j.csv")

    J = data_j.shape[0]
    logger.info(f"J={J}")
    prod2id = {i: i for i in range(J)}

    input_cust2id = input_id2cust = {i: i for i in range(I)}

    # configure data streamers
    experiment = modules.lib.read_yaml(f"{path_data}/{model}/experiment.yaml")
    model_module = importlib.import_module(f"modules.{model}")

    experiment["test_streamer_parameters"] = {
        "time_first": 100,
        "time_last": 100,
        "randomize": False,
        "history_length": experiment["global_streamer_parameters"]["history_length"],
        "full_history_pf": experiment["global_streamer_parameters"]["full_history_pf"],
    }

    # test data stream
    streamer_test = modules.data_streamer_v1.ProductChoiceDataStreamer(
        basket_data=baskets[baskets.i.isin(range(I))].reset_index(),
        action_data=actions[actions.i.isin(range(I))].reset_index(),
        cust2id=input_cust2id,
        id2cust=input_id2cust,
        prod2id=prod2id,
        **experiment["test_streamer_parameters"],
    )

    # load model and init trainer
    dnn_model = model_module.Model(
        J=streamer_test.J,
        T=streamer_test.history_length,
        pretrained=None,
        **experiment["params"],
    )

    dnn_model.load_state_dict(torch.load(file_sd, map_location="cpu", weights_only=True))
    _ = dnn_model.eval()

    dnn_trainer = modules.trainer.Trainer(
        model=dnn_model,
        dataset_train=None,
        experiment=experiment,
        save_yaml=False,
        makedir=False,
    )

    # extract probs
    prob = []

    # with discount
    for jn in range(J):
        streamer_test.reset_streamer(randomize=False)
        batch_size = streamer_test.num_training_samples
        labels, discounts, buycounts, frequencies, _ = streamer_test.get_batch(
            batch_size
        )

        assert np.sum(discounts) == 0
        discounts[:, jn] = discount

        labels = torch.from_numpy(labels).float()
        discounts = torch.from_numpy(discounts).float()
        buycounts = torch.from_numpy(buycounts).float()
        frequencies = torch.from_numpy(frequencies).float()
        if dnn_trainer.use_cuda:
            labels = labels.cuda()
            discounts = discounts.cuda()
            buycounts = buycounts.cuda()
            frequencies = frequencies.cuda()

        logits = dnn_trainer.Model(frequencies, discounts, buycounts)
        base_table = dnn_trainer._list_2_df(
            x=[torch.sigmoid(logits).detach().cpu().numpy()],
            streamer=streamer_test,
            value_name="prob",
        )

        # compute expected revenue
        base_table = base_table.merge(data_j[["j", "p_jc"]], on="j", how="left")
        base_table["price_paid"] = base_table["p_jc"] * (
            1 - discount * (base_table.j == jn)
        )
        base_table["coupon_j"] = jn
        base_table["discount"] = "discount"
        assert np.sum(base_table["price_paid"] != base_table["p_jc"]) == I

        prob.append(
            base_table[["i", "j", "coupon_j", "discount", "price_paid", "prob"]]
        )

    # without discount
    streamer_test.reset_streamer(randomize=False)
    batch_size = streamer_test.num_training_samples
    labels, discounts, buycounts, frequencies, _ = streamer_test.get_batch(batch_size)
    assert np.sum(discounts) == 0

    labels = torch.from_numpy(labels).float()
    discounts = torch.from_numpy(discounts).float()
    buycounts = torch.from_numpy(buycounts).float()
    frequencies = torch.from_numpy(frequencies).float()
    if dnn_trainer.use_cuda:
        labels = labels.cuda()
        discounts = discounts.cuda()
        buycounts = buycounts.cuda()
        frequencies = frequencies.cuda()

    logits = dnn_trainer.Model(frequencies, discounts, buycounts)
    base_table = dnn_trainer._list_2_df(
        x=[torch.sigmoid(logits).detach().cpu().numpy()],
        streamer=streamer_test,
        value_name="prob",
    )

    # compute expected revenue
    base_table = base_table.merge(data_j[["j", "p_jc"]], on="j", how="left")

    base_table["price_paid"] = base_table["p_jc"]
    base_table["coupon_j"] = 0
    base_table["discount"] = "no discount"
    assert np.sum(base_table["price_paid"] != base_table["p_jc"]) == 0

    prob.append(base_table[["i", "j", "coupon_j", "discount", "price_paid", "prob"]])

    # save
    total_prob = pd.concat(prob)
    path_output = f"{path_data}/prob_uplift"
    os.makedirs(path_output, exist_ok=True)
    total_prob.to_parquet(file_result)

    logger.info("□")


if __name__ == "__main__":

    args = modules.args.global_args("Discount simulation `dnn`")
    all_path_data = modules.lib.get_data_paths(args)

    for name, params in all_path_data.items():
        main(x=args.c, **params)
