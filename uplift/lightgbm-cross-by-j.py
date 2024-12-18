# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import tqdm
import gensim.models
import numpy as np
import pandas as pd
import lightgbm
from joblib import Parallel, delayed
from loguru import logger

import modules.args
import modules.lib
import data.word2vec
import modules.data_streamer_v1
import modules.dcm_dataset_cross


def main(x, emb, path_data, **kwargs):

    logger.info("Discount simulation `lightgbm-cross by j`")

    # load config
    config = modules.lib.read_yaml(x)
    config["path_data"] = path_data
    path_output = f"{path_data}/prob_uplift"
    config_prob = config["coupons"]
    I = config_prob["I"]
    discount = config_prob["discount"]
    logger.info(f"path_data={path_data}")
    logger.info(f"I={I:,}")
    logger.info(f"discount={discount}")
    label = "LightGBM_Cross_ByJ"
    label2 = "total_prob_lightgbm_cross_by_j"
    logger.info(f"label={label}")

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
    file_model = f"{path_data}/baselines/models/{label}.yaml"
    file_result = f"{path_output}/{label2}.parquet"
    if modules.lib.check_state(file_model, 1e-8, file_result, path_data):
        return 0
    os.makedirs(path_output, exist_ok=True)
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
    baskets["quantity"] = 1

    baseline_pf = pd.read_parquet(f"{path_data}/baselines/ProductPF.parquet")
    baseline_pf = baseline_pf[baseline_pf["t"] == 99].reset_index(drop=True)
    baseline_pf = baseline_pf[["i", "j", "phat"]].set_index("j")

    data_j = pd.read_csv(f"{path_data}/data_j.csv")

    J = data_j.shape[0]
    prod2id = {i: i for i in range(J)}

    input_cust2id = input_id2cust = {i: i for i in range(I)}

    # load (and normalize) embedding
    emb_raw = pd.read_parquet(f"{path_data}/{file_embedding}")
    emb_raw["j"] = emb_raw["j"].astype(int)
    if "c" in emb_raw:
        del emb_raw["c"]  # important <-- remove indices
    embedding = emb_raw.sort_values("j").set_index(["j"])
    assert embedding.shape[1] == config["w2v"]["size"]
    embedding_norm = embedding.values
    embedding_norm /= np.linalg.norm(embedding_norm, axis=1)[:, np.newaxis]

    # configure data streamers
    models = modules.lib.load_pickle(f"{path_data}/baselines/models/{label}.pickle")
    experiment = modules.lib.read_yaml(f"{path_data}/baselines/models/{label}.yaml")

    streamer_test = modules.data_streamer_v1.ProductChoiceDataStreamer(
        basket_data=baskets[baskets.i.isin(range(I))].reset_index(),
        action_data=actions[actions.i.isin(range(I))].reset_index(),
        prod2id=prod2id,
        cust2id=input_cust2id,
        id2cust=input_id2cust,
        full_history_pf=True,
        history_length=config["training"]["history_length"],
        time_first=100,
        time_last=100,
    )

    _, X_test, index_it_test = modules.dcm_dataset_cross.build_dcm_dataset(
        streamer=streamer_test,
        N=streamer_test.num_training_samples * J,
        batch_size=5_000,
        w2v_embedding=embedding_norm,
        bc_average_window_sizes=config["training"]["avg_windows"],
        randomize=False,
    )

    tmp1 = pd.DataFrame(streamer_test.user_time_pairs)
    tmp1.columns = ["i", "t"]
    tmp1["_tmp_key"] = 1
    tmp2 = pd.DataFrame({"j": list(range(streamer_test.J))})
    tmp2["_tmp_key"] = 1
    prediction = pd.merge(tmp1, tmp2, on="_tmp_key", how="left")
    del prediction["_tmp_key"]
    prediction["idx"] = list(range(prediction.shape[0]))
    idx_table = prediction[["j", "idx"]].set_index("j")
    prediction = prediction.merge(data_j, on="j", how="left")

    # extract probs
    logger.info(f"predict by product")

    def predict_jx(
        jn,
        X_test,
        discount,
        prediction,
        models,
        baseline_pf,
        streamer_test,
    ):

        # with discount
        x_input = X_test.copy()
        x_input[:, 3:253] = 0
        x_input[:, 3 + jn] = discount  # cross + own

        base_table = []
        for jx in range(J):
            idx_jx_test = np.arange(0, x_input.shape[0], J) + jx
            prediction_jx = prediction.iloc[idx_jx_test].reset_index(drop=True)
            assert np.all(prediction_jx.j == jx)

            if models[jx] == "none" or models[jx] == "error":
                prediction_jx["prob"] = (
                    baseline_pf.loc[(jx)].reset_index(drop=True)["phat"].values
                )
            else:
                prediction_jx["prob"] = models[jx].predict(x_input[idx_jx_test, :])
            prediction_jx["j"] = prediction_jx["j"].map(streamer_test.id2prod)
            prediction_jx["i"] = prediction_jx["i"].map(streamer_test.id2cust)
            base_table.append(prediction_jx)

        base_table = pd.concat(base_table).sort_values(["i", "t", "j"])

        # compute expected revenue
        base_table["price_paid"] = base_table["p_jc"] * (
            1 - discount * (base_table.j == jn)
        )

        base_table["coupon_j"] = jn
        base_table["discount"] = "discount"

        return base_table[["i", "j", "coupon_j", "discount", "price_paid", "prob"]]

    prob = Parallel(n_jobs=32)(
        delayed(predict_jx)(
            jn,
            X_test,
            discount,
            prediction,
            models,
            baseline_pf,
            streamer_test,
        )
        for jn in range(J)
    )

    # without discount
    x_input = X_test.copy()
    x_input[:, 3:253] = 0
    base_table = []
    for jx in range(J):
        idx_jx_test = np.arange(0, X_test.shape[0], J) + jx
        prediction_jx = prediction.iloc[idx_jx_test].reset_index(drop=True)
        assert np.all(prediction_jx.j == jx)

        if models[jx] == "none" or models[jx] == "error":
            prediction_jx["prob"] = (
                baseline_pf.loc[(jx)].reset_index(drop=True)["phat"].values
            )
        else:
            prediction_jx["prob"] = models[jx].predict(x_input[idx_jx_test, :])
        prediction_jx["j"] = prediction_jx["j"].map(streamer_test.id2prod)
        prediction_jx["i"] = prediction_jx["i"].map(streamer_test.id2cust)
        base_table.append(prediction_jx)

    base_table = pd.concat(base_table).sort_values(["i", "t", "j"])

    # compute expected revenue
    base_table["price_paid"] = base_table["p_jc"]
    base_table["coupon_j"] = 0
    base_table["discount"] = "no discount"

    prob.append(base_table[["i", "j", "coupon_j", "discount", "price_paid", "prob"]])

    # save
    total_prob = pd.concat(prob)
    total_prob.to_parquet(file_result)

    logger.info("□")


if __name__ == "__main__":

    args = modules.args.global_args("Discount simulation `lightgbm-cross by j`")
    all_path_data = modules.lib.get_data_paths(args)

    for name, params in all_path_data.items():
        if modules.lib.check_bootstrap_iter(params["seed_offset"], args.y, args.z):
            continue
        main(x=args.c, emb=args.emb, **params)
