# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import tqdm
import statsmodels.discrete.discrete_model as sm
import sklearn.metrics
import numpy as np
import pandas as pd
from loguru import logger

import modules.args
import modules.lib
import modules.data_streamer_v1
import modules.dcm_dataset_cross_base


def main(x, path_data, **kwargs):

    logger.info("Baseline `logit-cross by j`")

    # load config
    config = modules.lib.read_yaml(x)
    config["path_data"] = path_data
    logger.info(f"path_data={path_data}")
    os.makedirs(f"{path_data}/baselines", exist_ok=True)
    os.makedirs(f"{path_data}/baselines/models", exist_ok=True)
    I_train = config["logit"]["I_train"]

    # check state
    file_master = f"{path_data}/prediction_master_2.parquet"
    file_result = f"{path_data}/baselines/models/BinaryLogit_Cross_ByJ.yaml"
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
        "model_name": "BinaryLogit_Cross_ByJ",
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

    # load (and normalize) embedding
    embedding = (
        pd.read_parquet(f"{path_data}/w2v_embedding_j.parquet")
        .sort_values("j")
        .set_index(["c", "j"])  # important <-- remove indices
    )
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
    logger.info(
        f"number of training samples = {streamer_train.num_training_samples*J:,}"
    )
    y_train, X_train, index_it = modules.dcm_dataset_cross_base.build_dcm_dataset(
        streamer=streamer_train,
        N=streamer_train.num_training_samples * J,
        batch_size=5_000,
        w2v_embedding=embedding_norm,
        randomize=True,
        bc_average_window_sizes=config["training"]["avg_windows"],
    )
    assert len(y_train) == (index_it.shape[0] * J)
    assert X_train.shape[1] == (4 + len(config["training"]["avg_windows"]))
    if X_train[:, 1].sum() == 0:
        logger.warning("no discounts, remove discount column from data")
        remove_discounts = True
        X_train = X_train[:, [x for x in range(X_train.shape[1]) if x != 1]]
    else:
        remove_discounts = False
    assert np.sum(np.isnan(X_train)) == 0

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
    y_test, X_test, index_it_test = modules.dcm_dataset_cross_base.build_dcm_dataset(
        streamer=streamer_test,
        N=(streamer_test.num_training_samples * streamer_test.J),
        batch_size=5_000,
        w2v_embedding=embedding_norm,
        randomize=False,
        bc_average_window_sizes=config["training"]["avg_windows"],
    )
    if remove_discounts:
        X_test = X_test[:, [x for x in range(X_test.shape[1]) if x != 1]]
    assert X_test.shape[1] == (4 + len(config["training"]["avg_windows"]))
    pd.testing.assert_frame_equal(
        prediction[["i", "t"]].drop_duplicates().reset_index(drop=True),
        pd.DataFrame(streamer_test.user_time_pairs, columns=["i", "t"]),
    )
    pd.testing.assert_frame_equal(
        prediction[["i", "t"]].drop_duplicates().reset_index(drop=True), index_it_test
    )

    # train and predict by product
    logger.info(f"train/predict by product")
    models = {}
    predictions = []
    for jx in range(J):

        idx_jx = np.arange(0, len(y_train), J) + jx
        idx_jx_test = np.arange(0, len(y_test), J) + jx
        prediction_jx = prediction.iloc[idx_jx_test].reset_index(drop=True)
        assert np.all(prediction_jx.j == jx)

        if np.sum(y_train[idx_jx]) < 50:
            models[jx] = "none"
            prediction_jx["phat"] = np.NaN
        else:
            try:
                model = sm.Logit(y_train[idx_jx], X_train[idx_jx, :])
                result = model.fit(disp=0)
                models[jx] = result
                prediction_jx["phat"] = result.predict(X_test[idx_jx_test, :])
            except:
                models[jx] = "none"
                prediction_jx["phat"] = np.NaN

        prediction_jx["_debug_y"] = y_test[idx_jx_test].astype(int)
        prediction_jx["j"] = prediction_jx["j"].map(streamer_test.id2prod)
        prediction_jx["i"] = prediction_jx["i"].map(streamer_test.id2cust)
        predictions.append(prediction_jx)

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
    prediction_res.to_parquet(f"{path_data}/baselines/BinaryLogit_Cross_ByJ.parquet")
    modules.lib.dump_pickle(
        models, f"{path_data}/baselines/models/BinaryLogit_Cross_ByJ.pickle"
    )
    os.system(f"gzip {path_data}/baselines/models/BinaryLogit_Cross_ByJ.pickle")
    modules.lib.write_yaml(experiment_tracker, file_result)

    logger.info("□")


if __name__ == "__main__":

    args = modules.args.global_args("Baseline `logit-cross by j`")
    all_path_data = modules.lib.get_data_paths(args)

    for name, params in all_path_data.items():
        if modules.lib.check_bootstrap_iter(params["seed_offset"], args.y, args.z):
            continue
        main(x=args.c, **params)
