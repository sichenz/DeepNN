# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import re
import torch
import importlib
import numpy as np
import pandas as pd

from loguru import logger

import modules.args
import modules.lib
import modules.data_streamer_v1
import modules.data_loader
import modules.trainer


def main(
    file_config,
    pickle,
    path_data,
    model,
    suffix,
    overwrite,
    I=None,
    n_epoch=100,
    **kwargs,
):

    # config
    config = modules.lib.read_yaml(file_config)

    # experiment tracker
    experiment = modules.lib.read_yaml("dnn/config_model.yaml")
    experiment["status"] = "running"
    # overwrite model
    experiment["model"] = model
    # overwrite experiment_suffix
    experiment["experiment_suffix"] = suffix
    model_module = importlib.import_module(f"modules.{experiment['model']}")
    experiment["trainer"]["path"] = f"{path_data}/{model}{suffix}"
    logger.info(f"Starting model training")
    logger.info(f"path_data={path_data}")

    if os.path.isfile(
        f"{path_data}/{model}{suffix}/results/state_dict_{n_epoch-1:08d}.pt"
    ):
        logger.warning("results already exists, skip training...")
        return 0

    if os.path.isfile(f"{path_data}/{model}{suffix}/experiment.yaml"):
        logger.warning("training already running, skip training...")
        return 0

    os.makedirs(f"{path_data}/{model}{suffix}", exist_ok=True)
    file_experiment = f"{path_data}/{model}{suffix}/experiment.yaml"
    modules.lib.write_yaml(experiment, file_experiment)

    logger.info(f"path_result={path_data}/{model}{suffix}")
    logger.info(f"pickle={False}")
    logger.info(f"model={model}")
    logger.info(f"n_epoch={n_epoch}")

    # load data
    baskets = pd.read_parquet(f"{path_data}/baskets.parquet")
    if I is not None:
        I_train = I
    else:
        I_train = 100_000
    I_train = min(I_train, baskets.i.nunique())
    logger.info(f"I_train={I_train:,}")
    data_j = pd.read_csv(f"{path_data}/data_j.csv")
    J = data_j.shape[0]
    prod2id = {i: i for i in range(J)}
    T = config["training"]["history_length"]

    # training data stream and data loader
    experiment["global_streamer_parameters"] = {
        "history_length": T,
        "full_history_pf": config["training"]["full_history_pf"],
    }
    config_validation_streamer = {
        "time_first": config["training"]["time_last"],
        "time_last": config["training"]["time_last"],
        **experiment["global_streamer_parameters"],
    }
    dataset_dnn_validation = modules.data_loader.get_data_loader(
        c=config_validation_streamer,
        l="validation",
        p=path_data,
        ptmp="deprecated",
        prod2id=prod2id,
        I=I_train,
        do_pickle=False,
    )

    config_train_streamer = {
        "time_first": config["training"]["time_first"],
        "time_last": config["training"]["time_last"] - 1,  # last week for validation!
        **experiment["global_streamer_parameters"],
    }
    dataset_dnn_train = modules.data_loader.get_data_loader(
        c=config_train_streamer,
        l="train",
        p=path_data,
        ptmp="deprecated",
        prod2id=prod2id,
        I=I_train,
        do_pickle=False,
    )

    # seeds
    torch.manual_seed(501)
    np.random.seed(501)

    # model
    if experiment["pretrained_weights"] is not None:
        experiment["pretrained_weights"]["file"] = experiment["pretrained_weights"][
            "file"
        ].format(path_data=path_data)
    if overwrite is not None:
        if "w_conv_t2" in overwrite:
            experiment["params"]["K"] = 5
    dnn_model = model_module.Model(
        J=J,
        T=T,
        pretrained=experiment["pretrained_weights"],
        **experiment["params"],
    )

    # initialize trainer
    dnn_trainer = modules.trainer.Trainer(
        model=dnn_model,
        dataset_train=dataset_dnn_train,
        dataset_validation=dataset_dnn_validation,
        experiment=experiment,
    )

    # freeze weight
    freeze_weights = experiment["freeze_weights"]
    if freeze_weights is None:
        freeze_weights = []

    # manually manipulate weights (e.g., for nested model specifications)
    if overwrite is not None:
        if "w_conv_t2" in overwrite:
            print("overwrite w_conv_t")
            freeze_weights += ["w_conv_t"]
            # manipulate time embedding = fixed average
            _tmp_w_conv_t = dnn_trainer.Model.w_conv_t.data
            for i, w in enumerate(config["training"]["avg_windows"]):
                _tmp_w_conv_t[:, i] = _tmp_w_conv_t[:, i] * 0.0 + 1.0 / w
                _tmp_w_conv_t[w:, i] = 0
            dnn_trainer.Model.w_conv_t.data = _tmp_w_conv_t
        if "w_conv_t" in overwrite:
            print("overwrite w_conv_t")
            freeze_weights += ["w_conv_t"]
            # manipulate time embedding = fixed average
            dnn_trainer.Model.w_conv_t.data = dnn_trainer.Model.w_conv_t * 0.0 + 1.0 / T
        if "w_conv_j" in overwrite:
            print("overwrite w_conv_j")
            freeze_weights += ["w_conv_j", "w_conv_j_d", "w_conv_j_pf"]
            # manipulate bottleneck layer = identity transformation
            convj_identity = torch.eye(J)
            if dnn_trainer.use_cuda:
                convj_identity = convj_identity.cuda()
            dnn_trainer.Model.w_conv_j.data = convj_identity
            dnn_trainer.Model.w_conv_j_d.data = convj_identity
            dnn_trainer.Model.w_conv_j_pf.data = convj_identity

    # no discounts
    if not os.path.isfile(f"{path_data}/action.parquet"):
        logger.warning("no discounts, set discount weights to 0 and freeze weights")
        freeze_weights = [
            "w_conv_j_d",
            "w_conv_j_d2",
            "w_out_discount_cross",
            "w_out_discount",
        ]
        dnn_trainer.Model.w_conv_j_d.data = 0 * dnn_trainer.Model.w_conv_j_d.data
        dnn_trainer.Model.w_conv_j_d2.data = 0 * dnn_trainer.Model.w_conv_j_d2.data
        dnn_trainer.Model.w_out_discount.data = (
            0 * dnn_trainer.Model.w_out_discount.data
        )
        dnn_trainer.Model.w_out_discount_cross.data = (
            0 * dnn_trainer.Model.w_out_discount_cross.data
        )

    # logit
    baskets = baskets[(baskets["t"] > 70) & (baskets["t"] < 80)]
    tmp = baskets.groupby("j")[["i"]].count() / (
        baskets.i.nunique() * baskets.t.nunique()
    )
    tmp["logit"] = tmp.eval("log(i/(1-i))")
    assert np.max(np.abs(tmp["i"] - 1 / (1 + np.exp(-tmp["logit"])))) < 1e-12
    ff_out_b_init = torch.from_numpy(tmp[["logit"]].values[np.newaxis, :, :]).float()
    if dnn_trainer.use_cuda:
        ff_out_b_init = ff_out_b_init.cuda()
    dnn_trainer.Model.ff_out_b.data = ff_out_b_init

    # train
    dnn_trainer.train(
        n_epoch=n_epoch,
        batch_size=2048,
        batch_size_validation=min(len(dataset_dnn_validation), 4096),
        freeze_weights=freeze_weights,
    )

    # update experiment
    experiment["status"] = "done"
    modules.lib.write_yaml(experiment, file_experiment)

    logger.info("□")


if __name__ == "__main__":

    args = modules.args.global_args("Train DNN.")
    args.p = False
    # args.l = "scaling"
    all_path_data = modules.lib.get_data_paths(args, rerun=True)

    for name, params in all_path_data.items():
        main(
            file_config=args.c,
            pickle=args.p,
            **params,
        )
