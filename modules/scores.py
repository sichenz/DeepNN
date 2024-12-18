# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import torch
import scipy.special
import sklearn.metrics
import numpy as np
import pandas as pd

from loguru import logger

import modules.trainer


def loss(*args, **kwargs):
    return sklearn.metrics.log_loss(**kwargs)


def auc(*args, **kwargs):
    return sklearn.metrics.roc_auc_score(**kwargs)


def kl(p, phat, eps=1e-8):
    return np.mean(scipy.special.kl_div(phat + eps, p))


def r_squared(x, true, predicted):
    if true == predicted:
        return 1
    compare = x[[true, predicted]]
    return (
        1
        - ((compare[predicted] - compare[true]) ** 2).sum()
        / ((compare[true] - compare[true].mean()) ** 2).sum()
    )


def time_correlation(x, variable1, variable2, index_variables=["i", "j"], eps=1e-6):
    def get_matrix_variable(x, variable, index_variables):
        temp_df = x.pivot_table(index=index_variables, columns="t", values=[variable])
        temp_matrix = temp_df.values
        sums = temp_matrix.sum(axis=1)
        temp_matrix = temp_matrix - temp_matrix.mean(axis=1)[:, np.newaxis]
        return temp_matrix, sums, temp_df

    # turn data from long format (df) into wide format (matrix, each row is user-product
    # time series) normalize data
    xsub = x[set(index_variables + ["t", variable1, variable2])].copy()
    xsub["p"] = xsub["p"].values + np.random.uniform(0, eps, xsub.shape[0])
    xsub[variable2] = xsub[variable2].values + np.random.uniform(0, eps, xsub.shape[0])
    matrix_variable1, sums1, temp_df1 = get_matrix_variable(
        xsub, variable1, index_variables
    )
    matrix_variable2, sums2, temp_df2 = get_matrix_variable(
        xsub, variable2, index_variables
    )

    # manually compute correlation and covariance coefficients (for each user-product
    # combination)
    cor_denom = (
        matrix_variable2.shape[1]
        * matrix_variable1.std(axis=1)
        * matrix_variable2.std(axis=1)
    )
    # covs = (matrix_variable1 * matrix_variable2).sum(axis=1) /
    # (matrix_variable2.shape[1] - 1) mean_cov = np.mean(covs)
    cors = (matrix_variable1 * matrix_variable2).sum(axis=1) / cor_denom
    mean_cor = np.mean(cors)

    return mean_cor


def get_scores(x, x_pd, x_cd, x_ccd, x_tsd, x_tsnd, m, index_variables=["i", "j"]):
    logger.info(f"compute scores for for model {m}")
    return pd.DataFrame(
        {
            "model": [m],
            "kl-divergence": kl(x["p"], x[m]),
            "auc": auc(y_true=x["y"], y_score=x[m]),
            "log-loss": loss(y_true=x["y"], y_pred=x[m]),
            "log-loss_product_discount": -1
            if x_pd.shape[0] == 0
            else loss(y_true=x_pd["y"], y_pred=x_pd[m]),
            "log-loss_category_discount": -1
            if x_cd.shape[0] == 0
            else loss(y_true=x_cd["y"], y_pred=x_cd[m]),
            "log-loss_cross_category_discount": -1
            if x_ccd.shape[0] == 0
            else loss(y_true=x_ccd["y"], y_pred=x_ccd[m]),
            "time-correlation": time_correlation(
                x, "p", m, index_variables=index_variables
            ),
            "time-correlation_no_discount": time_correlation(
                x_tsnd, "p", m, index_variables=index_variables
            ),
        }
    )


def get_scores_minimal(x, x_tsnd, m, index_variables=["i", "j"]):
    logger.info(f"compute scores for for model {m}")
    return pd.DataFrame(
        {
            "model": [m],
            "kl-divergence": kl(x["p"], x[m]),
            "auc": auc(y_true=x["y"], y_score=x[m]),
            "log-loss": loss(y_true=x["y"], y_pred=x[m]),
            "time-correlation": time_correlation(
                x, "p", m, index_variables=index_variables
            ),
            "time-correlation_no_discount": time_correlation(
                x_tsnd, "p", m, index_variables=index_variables
            ),
        }
    )


def get_score_overview(
    epoch,
    model_module,
    path_data,
    model,
    suffix,
    streamer_test,
    experiment,
    prediction_master,
    actions,
):
    dnn_model_test = model_module.Model(
        J=streamer_test.J,
        T=streamer_test.history_length,
        pretrained=None,
        **experiment["params"],
    )
    dnn_model_test.load_state_dict(
        torch.load(f"{path_data}/{model}{suffix}/results/state_dict_{epoch:08d}.pt")
    )
    _ = dnn_model_test.eval()
    dnn_trainer_test = modules.trainer.Trainer(
        model=dnn_model_test,
        dataset_train=None,
        experiment=experiment,
    )

    pred = dnn_trainer_test.predict(streamer_test, 5_000)
    tmp = pred.merge(actions, on=["i", "j", "t"], how="left").merge(
        prediction_master, on=["i", "j", "t"]
    )
    tmp_discount = tmp[tmp["discount"].notnull()]
    kl = modules.scores.kl(tmp[["p"]].values, tmp[["phat"]].values)
    kld = modules.scores.kl(tmp_discount[["p"]].values, tmp_discount[["phat"]].values)
    ll = sklearn.metrics.log_loss(tmp[["y"]], tmp[["phat"]])
    lld = sklearn.metrics.log_loss(tmp_discount[["y"]], tmp_discount[["phat"]])
    rmse = sklearn.metrics.mean_squared_error(tmp[["p"]], tmp[["phat"]], squared=False)
    rmsed = sklearn.metrics.mean_squared_error(
        tmp_discount[["p"]], tmp_discount[["phat"]], squared=False
    )

    print(f"kl =   {kl:.4f}, {kld:.4f}")
    print(f"ll =   {ll:.4f}, {lld:.4f}")
    print(f"rmse = {rmse:.4f}, {rmsed:.4f}")
