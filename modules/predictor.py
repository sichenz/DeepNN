# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import torch
import importlib
from loguru import logger

import modules.lib
import modules.data_streamer_v1
import modules.trainer


def _predictor_core(
    path,
    model,
    suffix,
    epoch,
    prediction_master,
    baskets,
    actions,
    J,
    time_first,
    time_last,
    use_gpu=False,
):

    path_suffix = f"{path}/{model}{suffix}"
    if not os.path.isdir(path_suffix):
        return None

    # experiment
    model_module = importlib.import_module(f"modules.{model}")
    experiment = modules.lib.read_yaml(f"{path_suffix}/experiment.yaml")
    experiment["test_streamer_parameters"] = {
        "time_first": time_first,
        "time_last": time_last,
        "randomize": False,
        "history_length": experiment["global_streamer_parameters"]["history_length"],
        "full_history_pf": experiment["global_streamer_parameters"]["full_history_pf"],
    }

    experiment["prediction"] = {
        "path": experiment["trainer"]["path"],
        "epoch": epoch,
    }

    # test data stream
    streamer_test = modules.data_streamer_v1.ProductChoiceDataStreamer(
        basket_data=baskets,
        action_data=actions,
        prod2id={i: i for i in range(J)},
        **experiment["test_streamer_parameters"],
    )

    # load model and init trainer
    dnn_model = model_module.Model(
        J=streamer_test.J,
        T=streamer_test.history_length,
        pretrained=None,
        **experiment["params"],
    )

    if use_gpu:
        state_dict = torch.load(
            f"{path_suffix}/results/state_dict_{epoch:08d}.pt",
        )
    else:
        state_dict = torch.load(
            f"{path_suffix}/results/state_dict_{epoch:08d}.pt",
            map_location=lambda storage, loc: storage,
            weights_only=True,  # Explicitly use weights_only
        )

    dnn_model.load_state_dict(state_dict)
    _ = dnn_model.eval()

    dnn_trainer = modules.trainer.Trainer(
        model=dnn_model,
        dataset_train=None,
        experiment=experiment,
        save_yaml=False,
        makedir=False,
        use_gpu=use_gpu,
    )

    # prediction
    prediction = dnn_trainer.predict(x=streamer_test)
    res = prediction_master.merge(prediction, on=["i", "t", "j"], how="left")
    if res["phat"].isnull().sum() > 0:
        logger.warning("%d NA values in predictions" % res["phat"].isnull().sum())

    return res
