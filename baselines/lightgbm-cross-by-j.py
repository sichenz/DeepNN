# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import gc
import tqdm
import lightgbm
import sklearn.metrics
import gensim.models
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

import modules.args
import modules.lib
import modules.data_streamer_v1
import modules.dcm_dataset_cross_j


def main(x, emb, path_data, **kwargs):

    logger.info("Baseline `lightgbm-cross by j`")

    # load config
    config = modules.lib.read_yaml(x)
    config["path_data"] = path_data
    logger.info(f"path_data={path_data}")
    os.makedirs(f"{path_data}/baselines", exist_ok=True)
    os.makedirs(f"{path_data}/baselines/models", exist_ok=True)
    I_train = config["lightgbm"]["I_train"]
    label = "LightGBM_Cross_ByJ"
    n_cores = config["n_cores"]["lightgbm"]
    logger.info(f"label={label}")
    logger.info(f"n_cores={n_cores}")

    if emb == "w2v":
        file_embedding = "w2v_embedding_j.parquet"
        result_suffix = ""
    elif emb == "glove":
        file_embedding = "glove/glove-embedding.parquet"
        result_suffix = "_glove"
    else:
        raise Exception("embedding type not supported")
    logger.info(f"embedding={emb}")

    # check state
    file_master = f"{path_data}/prediction_master_2.parquet"
    file_result = f"{path_data}/baselines/models/{label}{result_suffix}.yaml"
    if modules.lib.check_state(file_master, 1, file_result, path_data):
        return 0
    modules.lib.touch(file_result)

    # load data
    prediction = pd.read_parquet(file_master, columns=["i", "t", "j"]).sort_values(
        ["i", "t", "j"]
    )
    baseline_pf = pd.read_parquet(f"{path_data}/baselines/ProductPF.parquet")
    baskets = pd.read_parquet(f"{path_data}/baskets.parquet")
    baskets["quantity"] = 1.0
    I_train = min(I_train, baskets.i.nunique())
    file_actions = f"{path_data}/action.parquet"
    if os.path.isfile(file_actions):
        actions = pd.read_parquet(file_actions)
    else:
        logger.warning("no discounts")
        actions = pd.DataFrame({"i": [0], "j": 0, "t": 88, "discount": 0})
    data_j = pd.read_csv(f"{path_data}/data_j.csv")
    J = data_j.shape[0]
    prod2id = {i: i for i in range(J)}
    logger.info(f"I_train={I_train:,}")

    # experiment tracker
    logger.info(f"setup streamer")
    experiment_tracker = {
        "experiment": "dcm",
        "model_name": label,
    }

    experiment_tracker["model"] = {
        "path_embedding": f"{path_data}/gensim",
    }

    experiment_tracker["training_streamer_parameters"] = {
        "time_first": config["training"]["time_first"],
        "time_last": config["training"]["time_last"],
        "history_length": config["training"]["history_length"],
        "full_history_pf": True,
    }

    experiment_tracker["test_streamer_parameters"] = {
        "time_first": config["test_set"]["t_start"],
        "time_last": config["test_set"]["t_end"],
        "history_length": experiment_tracker["training_streamer_parameters"][
            "history_length"
        ],
        "full_history_pf": experiment_tracker["training_streamer_parameters"][
            "full_history_pf"
        ],
    }

    experiment_tracker["model_parameters"] = {
        "learning_rate": 0.02,
        "num_leaves": 25,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": "binary_logloss",
        "verbose": -1,
    }

    # load (and normalize) embedding
    emb_raw = pd.read_parquet(f"{path_data}/{file_embedding}")
    emb_raw["j"] = emb_raw["j"].astype(int)
    if "c" in emb_raw:
        del emb_raw["c"]  # important <-- remove indices
    embedding = emb_raw.sort_values("j").set_index(["j"])
    assert embedding.shape[1] == config["w2v"]["size"]
    embedding_norm = embedding.values
    embedding_norm /= np.linalg.norm(embedding_norm, axis=1)[:, np.newaxis]

    # training streamer
    np.random.seed(501)
    streamer_train = modules.data_streamer_v1.ProductChoiceDataStreamer(
        basket_data=baskets[baskets.i.isin(range(I_train))].reset_index(),
        action_data=actions[actions.i.isin(range(I_train))].reset_index(),
        prod2id=prod2id,
        **experiment_tracker["training_streamer_parameters"],
    )

    # prediction streamer
    streamer_test = modules.data_streamer_v1.ProductChoiceDataStreamer(
        basket_data=baskets[
            baskets.i.isin(range(config["test_set"]["I"]))
        ].reset_index(),
        action_data=actions[
            actions.i.isin(range(config["test_set"]["I"]))
        ].reset_index(),
        prod2id=prod2id,
        **experiment_tracker["test_streamer_parameters"],
    )

    # train and predict by product
    logger.info(f"train/predict by product")

    def train_jx(
        jx,
        streamer_train,
        embedding_norm,
        config,
        streamer_test,
        prediction,
        experiment,
    ):

        y_train, X_train, index_it = modules.dcm_dataset_cross_j.build_dcm_dataset(
            streamer=streamer_train,
            jx=jx,
            batch_size=5_000,
            w2v_embedding=embedding_norm,
            bc_average_window_sizes=config["training"]["avg_windows"],
            randomize=True,
        )
        gc.collect()
        assert len(y_train) == index_it.shape[0]
        assert X_train.shape[1] == 3 + J + len(config["training"]["avg_windows"])

        y_test, X_test, index_it_test = modules.dcm_dataset_cross_j.build_dcm_dataset(
            streamer=streamer_test,
            jx=jx,
            batch_size=5_000,
            w2v_embedding=embedding_norm,
            bc_average_window_sizes=config["training"]["avg_windows"],
            randomize=False,
        )
        pd.testing.assert_frame_equal(
            prediction[["i", "t"]].drop_duplicates().reset_index(drop=True),
            pd.DataFrame(streamer_test.user_time_pairs, columns=["i", "t"]),
        )
        pd.testing.assert_frame_equal(
            prediction[["i", "t"]].drop_duplicates().reset_index(drop=True),
            index_it_test,
        )
        gc.collect()

        idx_jx_test = np.arange(0, prediction.shape[0], J) + jx
        prediction_jx = prediction.iloc[idx_jx_test].reset_index(drop=True)
        assert np.all(prediction_jx.j == jx)
        prediction_jx["_debug_y"] = y_test.astype(int)

        n_val = len(y_train) // 10
        lgbm_data_train = lightgbm.Dataset(X_train[n_val:, :], label=y_train[n_val:])
        lgbm_data_validation = lightgbm.Dataset(
            X_train[:n_val, :], label=y_train[:n_val]
        )
        bst = lightgbm.train(
            experiment_tracker["model_parameters"],
            train_set=lgbm_data_train,
            num_boost_round=500,
            valid_sets=[lgbm_data_train, lgbm_data_validation],
            early_stopping_rounds=10,
            verbose_eval=False,
        )
        prediction_jx["phat"] = bst.predict(X_test)
        gc.collect()

        prediction_jx["j"] = prediction_jx["j"].map(streamer_test.id2prod)
        prediction_jx["i"] = prediction_jx["i"].map(streamer_test.id2cust)

        return bst, prediction_jx

    parallel_out = Parallel(n_jobs=n_cores)(
        delayed(train_jx)(
            jx,
            streamer_train,
            embedding_norm,
            config,
            streamer_test,
            prediction,
            experiment_tracker,
        )
        for jx in range(J)
    )
    models = [x[0] for x in parallel_out]
    predictions = [x[1] for x in parallel_out]

    # combine results
    prediction_res = pd.concat(predictions).sort_values(["i", "t", "j"])
    assert np.all(
        prediction_res[["i", "t", "j"]].values == prediction[["i", "t", "j"]].values
    )

    # fill NaNs with frequency baseline
    baseline_pf = baseline_pf.sort_values(["i", "t", "j"])
    assert np.all(
        prediction_res[["i", "t", "j"]].values == baseline_pf[["i", "t", "j"]].values
    )
    prediction_res["phat"].fillna(baseline_pf["phat"], inplace=True)

    # out-of-sample evaluation
    test_auc = sklearn.metrics.roc_auc_score(
        prediction_res["_debug_y"].values, prediction_res["phat"].values
    )
    test_loss = sklearn.metrics.log_loss(
        prediction_res["_debug_y"].values, prediction_res["phat"].values
    )
    logger.info(f"auc(test)={test_auc:.4f}")
    logger.info(f"loss(test)={test_loss:.6f}")

    # save
    logger.info(f"save model and predictions")
    prediction_res.to_parquet(f"{path_data}/baselines/{label}{result_suffix}.parquet")
    modules.lib.dump_pickle(
        models, f"{path_data}/baselines/models/{label}{result_suffix}.pickle"
    )
    modules.lib.write_yaml(experiment_tracker, file_result)

    logger.info("□")


if __name__ == "__main__":

    args = modules.args.global_args("Baseline `lightgbm-cross by j`")
    all_path_data = modules.lib.get_data_paths(args)

    for name, params in all_path_data.items():
        if modules.lib.check_bootstrap_iter(params["seed_offset"], args.y, args.z):
            continue
        main(x=args.c, emb=args.emb, **params)
