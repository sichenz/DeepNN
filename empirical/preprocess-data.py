# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import gc
import itertools
import os

import modules.args
import modules.lib
import numpy as np
import pandas as pd
import sklearn.decomposition
from loguru import logger

if __name__ == "__main__":

    logger.info("Preprocess data for the empirical application")

    # args
    args = modules.args.global_args("Preprocess data for the empirical application")
    args.c = "configs/config-empirical.yaml"
    logger.info(f"config = {args.c}")

    # config
    config = modules.lib.read_yaml(args.c)
    path_data_raw = os.path.expanduser(config["path_data_raw"])
    path_data = os.path.expanduser(config["path_data"])
    logger.info(f"path_data_raw = {path_data_raw}")
    logger.info(f"path_data = {path_data}")

    # load data
    data_j = pd.read_csv(f"{path_data_raw}/data_j.csv")
    baskets_raw = pd.read_csv(f"{path_data_raw}/lc_data.csv")
    actions_raw = pd.read_csv(f"{path_data_raw}/coupon_data.csv")
    w2v_embedding = pd.read_csv(f"{path_data_raw}/category_brand_vectors.csv")
    brand_master = pd.read_csv(f"{path_data_raw}/brand_map.csv")
    raw_lc = pd.read_parquet(
        f"{path_data_raw}/loyalty-card-data.parquet",
        columns=["product_id", "price"],
    )

    # reduce lc (only keep data needed for price computation)
    raw_lc = raw_lc[
        raw_lc["product_id"].isin(brand_master["product"].unique())
    ].reset_index(drop=True)
    raw_lc = raw_lc.groupby("product_id").price.agg(["mean", "count"])
    raw_lc = raw_lc.reset_index()
    raw_lc.rename(columns={"product_id": "product"}, inplace=True)
    gc.collect()

    # basket data
    baskets = baskets_raw.copy()
    # build new consumer ID (0, ..., # users), mimic synthetic data
    baskets["i"] = baskets.groupby(["user"]).grouper.group_info[0]
    baskets["quantity"] = 1.0
    baskets["article_text"] = baskets["j"]
    del baskets["j"]
    del baskets["y"]
    baskets = baskets[baskets.article_text.isin(data_j.article_text)]
    baskets = baskets.merge(
        data_j[["article_text", "j"]], on="article_text", how="left"
    )

    # user map
    user_map = baskets[["user", "i"]].drop_duplicates().sort_values("i")

    # actions
    actions = actions_raw.copy()
    actions = actions[actions["user"].isin(baskets.user)]
    actions["article_text"] = actions["j"]
    del actions["j"]
    assert np.all(actions[["article_text"]].isin(data_j.article_text.unique()))
    actions = actions.merge(
        data_j[["article_text", "j"]], on="article_text", how="left"
    )
    # add new consumer ID (0, ..., # of users), mimic synthetic data
    actions = actions.merge(user_map, on="user", how="left")

    # tests
    assert np.all(baskets["i"].notnull())
    assert np.all(baskets["j"].notnull())
    assert np.all(baskets["t"].notnull())
    assert np.all(actions["i"].notnull())
    assert np.all(actions["j"].notnull())
    assert np.all(actions["t"].notnull())
    assert np.all(actions.j.isin(baskets.j.unique()))
    assert np.all(actions.i.isin(baskets.i.unique()))
    assert np.all(actions.t.isin(baskets.t.unique()))
    gc.collect()

    # prune consumers
    # only consider consumers with at least 15 observations
    n_obs_required = 15
    t_window_start = (
        config["training"]["time_first"] - config["training"]["history_length"]
    )
    t_window_end = config["training"]["t_test_end"]
    baskets_tmp = baskets[
        (baskets["t"] >= t_window_start) & (baskets["t"] <= t_window_end)
    ]
    n_obs_i = baskets_tmp.groupby("i")[["t"]].nunique()
    n_obs_i_pruned = n_obs_i[n_obs_i["t"] >= n_obs_required]
    eligible_is = sorted(n_obs_i_pruned.reset_index().i.unique())
    logger.info(f"len(eligible_is) = {len(eligible_is):,}")
    baskets_pruned = baskets[baskets.i.isin(eligible_is)].reset_index(drop=True)
    actions_pruned = actions[
        (actions.i.isin(eligible_is)) & (actions.j.isin(baskets.j.unique()))
    ].reset_index(drop=True)

    # product map for aggregating products
    quantity_j = (
        baskets_pruned.groupby(["j", "article_text"], as_index=False)
        .quantity.sum()
        .merge(data_j, on=["j", "article_text"], how="left")
    )
    # products labelled `*--other` end up in other brand by setting sales to -1
    quantity_j.loc[quantity_j.article_text.str.match(r".*--other$"), "quantity"] = -1
    quantity_j = quantity_j.sort_values(
        ["quantity", "article_text"], ascending=[False, True]
    )
    assert np.all(quantity_j.category.notnull())
    quantity_j["j_counter"] = quantity_j.groupby("category").cumcount()
    quantity_j["article_text_2"] = quantity_j["article_text"].copy()
    quantity_j.loc[quantity_j["j_counter"] >= 9, "article_text_2"] = (
        quantity_j.loc[quantity_j["j_counter"] >= 9, "category"] + "--other"
    )
    assert quantity_j.groupby("category").article_text_2.nunique().max() <= 10
    product_id_map = quantity_j[
        ["article_text", "j", "article_text_2", "quantity"]
    ].sort_values("j")
    product_id_map["j_2"] = product_id_map.groupby("article_text_2").ngroup()
    logger.info(f"J: {product_id_map.j.nunique()} -> {product_id_map.j_2.nunique()}")

    # baskets
    baskets_new = (
        baskets_pruned.merge(product_id_map, on="j", how="left")[
            ["user", "i", "t", "j_2", "article_text_2"]
        ]
        .drop_duplicates()
        .rename(columns={"j_2": "j", "article_text_2": "article_text"})
    )
    assert baskets_new[baskets_new["article_text"].isnull()].shape[0] == 0
    baskets_new["quantity"] = 1.0
    logger.info(f"nrow(baskets): {baskets.shape[0]:,} -> {baskets_new.shape[0]:,}")

    # actions
    actions_new = (
        actions_pruned.merge(product_id_map, on="j", how="left")
        .groupby(["user", "t", "i", "j_2", "article_text_2"], as_index=False)
        .discount.max()
        .rename(columns={"j_2": "j", "article_text_2": "article_text"})
    )
    logger.info(f"nrow(actions): {actions.shape[0]:,} -> {actions_new.shape[0]:,}")

    # build prediction master
    np.random.seed(501)
    test_is = sorted(np.random.choice(eligible_is, 1_000, replace=False))
    test_js = sorted(baskets_new.j.unique())
    t_test_start = config["training"]["t_test_start"]
    t_test_end = config["training"]["t_test_end"]
    test_ts = np.arange(t_test_start, t_test_end + 1, 1)
    prediction_master = pd.DataFrame(
        np.array([x for x in itertools.product(test_is, test_js, test_ts)]),
        columns=["i", "j", "t"],
    )
    test_is_baskets = baskets_new[baskets_new.i.isin(prediction_master["i"])]
    test_is_baskets_red = test_is_baskets[["i", "j", "t"]].drop_duplicates()
    test_is_baskets_red["y"] = 1
    prediction_master = prediction_master.merge(
        test_is_baskets_red, on=["i", "j", "t"], how="left"
    )
    prediction_master["y"] = prediction_master["y"].fillna(0).astype(int)

    # w2v embedding
    w2v_embedding.rename(columns={"j": "article_text"}, inplace=True)
    w2v_embedding_new = (
        data_j[["j", "article_text"]]
        .merge(w2v_embedding, on="article_text")
        .merge(product_id_map[["j", "j_2", "quantity"]], on="j")
        .drop(columns=["j", "article_text"])
        .rename(columns={"j_2": "j"})
        .sort_values("quantity", ascending=False)
        .groupby("j")
        .head(1)
        .drop(columns=["quantity"])
        .set_index("j")
    )
    w2v_pca = sklearn.decomposition.PCA(n_components=config["training"]["L"])
    w2v_embedding_norm = w2v_pca.fit_transform(w2v_embedding_new.values)
    w2v_embedding_norm /= np.linalg.norm(w2v_embedding_norm, axis=1)[:, np.newaxis]
    w2v_embedding_new_norm = pd.DataFrame(w2v_embedding_norm, w2v_embedding_new.index)
    w2v_embedding_new_norm.columns = [str(x) for x in w2v_embedding_new_norm.columns]
    w2v_embedding_new_norm = w2v_embedding_new_norm.sort_index()

    # save
    os.makedirs(path_data, exist_ok=True)
    baskets_new.to_parquet(f"{path_data}/baskets.parquet")
    actions_new.to_parquet(f"{path_data}/action.parquet")
    product_id_map.to_parquet(f"{path_data}/product_id_map.parquet")
    prediction_master.to_parquet(f"{path_data}/prediction_index.parquet")
    w2v_embedding_new_norm.reset_index().to_parquet(
        f"{path_data}/w2v_embedding_new_norm.parquet"
    )
