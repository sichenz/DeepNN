# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os

import modules.args
import modules.lib
import numpy as np
import pandas as pd
from loguru import logger

if __name__ == "__main__":

    # args
    args = modules.args.global_args("table 5")
    args.c = "configs/config-empirical.yaml"
    config = modules.lib.read_yaml(args.c)
    config_train = config["training"]
    path_data_raw = os.path.expanduser(config["path_data_raw"])
    path_data = os.path.expanduser(config["path_data"])
    path_results = os.path.expanduser(config["path_results"])
    logger.info(f"path_data = {path_data}")
    logger.info(f"path_results = {path_results}")
    model = "model_010"
    os.makedirs(path_results, exist_ok=True)

    # data
    baskets = pd.read_parquet(f"{path_data}/baskets.parquet")
    actions = pd.read_parquet(f"{path_data}/action.parquet")
    actions_raw = pd.read_csv(f"{path_data_raw}/coupon_data.csv")
    actions_raw = actions_raw[actions_raw["user"].isin(baskets["user"])]
    actions_baskets = actions.merge(baskets, how="left", on=["i", "j", "t"])
    n_coup_i = (
        baskets[["i"]]
        .drop_duplicates()
        .merge(actions.groupby("i").j.count(), on="i", how="left")
    )
    n_coup_i.j.fillna(0, inplace=True)
    assert n_coup_i.j.mean() == actions.shape[0] / baskets.i.nunique()

    # data statistics
    n_weeks = config_train["t_test_end"] - (
        config_train["time_first"] - config_train["history_length"]
    )
    redemption_rate = actions_baskets["quantity"].notnull().sum() / actions_raw.shape[0]
    redemption_rate_se = actions_baskets["quantity"].notnull().std() / np.sqrt(
        actions_raw.shape[0]
    )

    out = pd.DataFrame(
        {
            "variable": [
                "# of users",
                "# of weeks",
                "# of brands",
                "# of stores",
                "# of coupons",
                "Avg. number of coupons per customer",
                "Discount range",
                "Redemption rate (SE)",
                "Avg. discount",
                "# of baskets",
                "# of montrs",
                "# of produts per basket",
            ],
            "value": [
                f"{baskets.i.nunique():,}",
                n_weeks,
                baskets.j.nunique(),
                155,
                f"{actions.shape[0]:,}",
                f"{n_coup_i.j.mean():.2f} ({n_coup_i.j.std():.2f})",
                "[5%, 50%]",
                f"{redemption_rate:.2%} ({redemption_rate_se:.2%})",
                f"{actions.discount.mean():.1%} ({actions.discount.std():.1%})",
                "73,048,605",
                12,
                4.91,
            ],
        }
    )

    out.to_html(f"{path_results}/paper/table_5.html")
