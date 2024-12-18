# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

# we simulate a data set with less discounts to simplify exposition
import importlib
import os

import modules.args
import modules.data_streamer_v1
import modules.dcm_dataset_cross_base
import modules.dcm_dataset_cross_j
import modules.lib
import modules.predictor
import numpy as np
import pandas as pd
import torch

lib = importlib.import_module("data.build-prediction-master")


if __name__ == "__main__":

    # setup
    args = modules.args.global_args("Figure 2 (Data)")
    config = modules.lib.read_yaml(args.c)
    path_data = os.path.expandvars(f"{config['path_data']}_009")
    # skip figure 2 if data set 9 is not available
    if not os.path.isdir(path_data):
        exit()

    # manual input
    I = 2_000
    t_coupon = 96
    j_own = 68
    path_out = f"{path_data}/figure-2"
    os.makedirs(path_out, exist_ok=True)

    # load data
    data_j = pd.read_csv(f"{path_data}/data_j.csv")
    action = pd.read_parquet(f"{path_data}/action.parquet")
    baskets = pd.read_parquet(f"{path_data}/baskets.parquet")
    # skip figure 2 if simulation config does not match paper
    if not baskets["i"].nunique() == 100_000:
        exit()
    baskets["y"] = 1
    gym = modules.lib.load_gzip_pickle(f"{path_data}/gym0_light.pickle.gz")

    # action
    action_90 = pd.DataFrame(range(I), columns=["i"])
    action_90["j"] = j_own
    action_90["discount"] = 0.3
    action_90["nc"] = 0
    action_90["t"] = t_coupon

    # build prediction master
    baskets_90, data_ic, data_ij = lib.extract_data_from_gym(
        x=gym,
        baskets=baskets,
        set_is=range(I),
        target_week=90,
        W=10,
        action=action_90,
    )
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
    del baskets["y"]
    baskets_mxl = baskets[baskets.t < 90].append(baskets_90)
    actions_mxl = action[action.t < 90].append(action_90)

    # save data for MXL
    prediction_master.to_parquet(f"{path_out}/prediction_master_2.parquet")
    baskets_mxl.to_parquet(f"{path_out}/baskets.parquet")
    actions_mxl.to_parquet(f"{path_out}/action.parquet")
    data_j.to_csv(f"{path_out}/data_j.csv", index=False)

    # DNN
    baskets_dnn = baskets_mxl.copy()
    actions_dnn = actions_mxl.copy()
    baskets_dnn["quantity"] = 1.0
    J = data_j.shape[0]
    prod2id = {i: i for i in range(J)}

    torch.manual_seed(501)
    np.random.seed(501)

    users = prediction_master.i.unique()
    baskets_users = baskets_dnn[baskets_dnn["i"].isin(users)].reset_index(drop=True)
    actions_users = actions_dnn[actions_dnn["i"].isin(users)].reset_index(drop=True)

    dnn_pred = modules.predictor._predictor_core(
        path=path_data,
        model="model_010",
        suffix="",
        epoch=99,
        prediction_master=prediction_master,
        baskets=baskets_users,
        actions=actions_users,
        J=J,
        time_first=90,
        time_last=99,
        use_gpu=False,
    )

    dnn_pred = dnn_pred[["i", "j", "t", "phat"]]
    dnn_pred.to_parquet(f"{path_out}/pred_dnn.parquet")
    del baskets_dnn, actions_dnn, baskets_users, actions_users

    # BINARY LOGIT
    baskets_logit = baskets_mxl.copy()
    actions_logit = actions_mxl.copy()
    baskets_logit["quantity"] = 1.0
    baseline_pf = pd.read_parquet(f"{path_data}/baselines/ProductPF.parquet")
    embedding = (
        pd.read_parquet(f"{path_data}/w2v_embedding_j.parquet")
        .sort_values("j")
        .set_index(["c", "j"])
    )
    assert embedding.shape[1] == 30
    embedding_norm = embedding.values
    embedding_norm /= np.linalg.norm(embedding_norm, axis=1)[:, np.newaxis]

    experiment_tracker = {
        "experiment": "dcm",
        "model_name": "BinaryLogit_Cross_ByJ",
    }
    experiment_tracker["test_streamer_parameters"] = {
        "time_first": 90,
        "time_last": 99,
        "history_length": 30,
        "full_history_pf": True,
    }

    streamer_test = modules.data_streamer_v1.ProductChoiceDataStreamer(
        basket_data=baskets_logit[baskets_logit.i.isin(range(I))].reset_index(),
        action_data=actions_logit[actions_logit.i.isin(range(I))].reset_index(),
        prod2id=prod2id,
        **experiment_tracker["test_streamer_parameters"],
    )

    y_test, X_test, index_it_test = modules.dcm_dataset_cross_base.build_dcm_dataset(
        streamer=streamer_test,
        N=(streamer_test.num_training_samples * streamer_test.J),
        batch_size=5_000,
        w2v_embedding=embedding_norm,
        randomize=False,
        bc_average_window_sizes=[1, 3, 5, 15, 30],
    )

    models = modules.lib.load_gzip_pickle(
        f"{path_data}/baselines/models/BinaryLogit_Cross_ByJ.pickle.gz"
    )

    predictions = []
    prediction = prediction_master[["i", "t", "j"]].sort_values(["i", "t", "j"])
    for jx in range(J):
        idx_jx_test = np.arange(0, len(y_test), J) + jx
        prediction_jx = prediction.iloc[idx_jx_test].reset_index(drop=True)
        assert np.all(prediction_jx.j == jx)
        if models[jx] == "none" or models[jx] == "error":
            prediction_jx["prob"] = (
                baseline_pf.loc[(jx)].reset_index(drop=True)["phat"].values
            )
        else:
            prediction_jx["prob"] = models[jx].predict(X_test[idx_jx_test, :])
        prediction_jx["_debug_y"] = y_test[idx_jx_test].astype(int)
        prediction_jx["j"] = prediction_jx["j"].map(streamer_test.id2prod)
        prediction_jx["i"] = prediction_jx["i"].map(streamer_test.id2cust)
        predictions.append(prediction_jx)

    prediction_res = pd.concat(predictions).sort_values(["i", "t", "j"])
    prediction_res["phat"] = prediction_res.prob
    prediction_res[["i", "j", "t", "phat"]].to_parquet(f"{path_out}/pred_logit.parquet")

    # LIGHTGBM
    input_cust2id = input_id2cust = {i: i for i in range(I)}
    models = modules.lib.load_pickle(
        f"{path_data}/baselines/models/LightGBM_Cross_ByJ.pickle"
    )

    streamer_test = modules.data_streamer_v1.ProductChoiceDataStreamer(
        basket_data=baskets_logit[baskets_logit.i.isin(range(I))].reset_index(),
        action_data=actions_logit[actions_logit.i.isin(range(I))].reset_index(),
        prod2id=prod2id,
        **experiment_tracker["test_streamer_parameters"],
    )

    predictions = []
    for jx in range(J):
        y_test, X_test, index_it_test = modules.dcm_dataset_cross_j.build_dcm_dataset(
            streamer=streamer_test,
            jx=jx,
            batch_size=5_000,
            w2v_embedding=embedding_norm,
            bc_average_window_sizes=[1, 3, 5, 15, 30],
            randomize=False,
        )
        idx_jx_test = np.arange(0, prediction.shape[0], J) + jx
        prediction_jx = prediction.iloc[idx_jx_test].reset_index(drop=True)
        assert np.all(prediction_jx.j == jx)
        prediction_jx["_debug_y"] = y_test.astype(int)
        prediction_jx["phat"] = models[jx].predict(X_test)
        prediction_jx["j"] = prediction_jx["j"].map(streamer_test.id2prod)
        prediction_jx["i"] = prediction_jx["i"].map(streamer_test.id2cust)
        predictions.append(prediction_jx)

    prediction_res = pd.concat(predictions).sort_values(["i", "t", "j"])
    assert prediction_res.phat.isnull().sum() == 0
    prediction_res[["i", "j", "t", "phat"]].to_parquet(
        f"{path_out}/pred_lightgbm.parquet"
    )
