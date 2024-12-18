# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import numpy as np
import pandas as pd
from loguru import logger


def build_dcm_dataset(
    streamer,
    N,
    data_j,
    batch_size,
    w2v_embedding,
    bc_average_window_sizes,
    randomize=True,
):
    """
    [:, 0]  intercept
    [:, 1]  frequencies
    [:, 2]  cosine_similarity
    [:, J]  discounts_cross
    [:, W]  bc_average_features
    [:, C]  discounts_cross (category level)
    [:, W]  bc_average_features (category level)
    """

    if False:
        streamer = streamer_test
        N = streamer_test.num_training_samples * J
        batch_size = 5_000
        w2v_embedding = w2v_embedding_norm
        bc_average_window_sizes = config["training"]["avg_windows"]
        randomize = False

    logger.info("build dcm dataset cat cross")

    streamer.reset_streamer(randomize=randomize)
    J = streamer.J
    T = streamer.history_length
    C = data_j.c.nunique()

    y_list = []
    x_list = []
    idx_list = []
    n_samples = 0
    needed_samples = N

    index_it = pd.DataFrame(streamer.shuffled_user_time_pairs, columns=["i", "t"])

    while needed_samples > 0:

        if len(streamer.user_time_pairs_cache) == 0:
            raise Exception("streamer empty and more samples needed")
        tmp = streamer.get_batch(min([len(streamer.user_time_pairs_cache), batch_size]))
        idx_list.append(
            pd.DataFrame(
                {
                    "i": tmp[4][0],
                    "t": tmp[4][1],
                }
            )
        )

        # aggregation J -> C
        aggregation_j_c = np.zeros((J, C))
        aggregation_j_c[(data_j.j, data_j.c)] = 1

        # build y
        y = tmp[0].flatten()  # "by row"
        # assert np.all(np.arange(0,12).reshape((3,4)).flatten() == np.arange(0,12))

        # build x
        discounts_cross = tmp[1][np.repeat(range(tmp[1].shape[0]), J), :]

        frequencies = tmp[3].flatten()

        # cs
        purchase_history = tmp[2].swapaxes(1, 2).reshape(-1, T)
        weights_buycounts_window = np.sum(tmp[2], axis=1)
        index_0 = np.sum(weights_buycounts_window, axis=1) == 0
        if len(index_0) > 0:
            #    logger.warning(f'fill zero weights_buycounts_window')
            weights_buycounts_window[index_0, :] = np.sum(
                weights_buycounts_window, axis=0
            )
        weights_buycounts_window /= np.sum(weights_buycounts_window, axis=1)[
            :, np.newaxis
        ]
        user_vector = np.dot(weights_buycounts_window, w2v_embedding)
        user_vector_norm = (
            user_vector / np.linalg.norm(user_vector, axis=1)[:, np.newaxis]
        )
        cosine_similarity = np.dot(user_vector_norm, w2v_embedding.T).flatten()

        # average window buycounts
        bc_average_features = np.zeros((len(y), len(bc_average_window_sizes)))
        for i, w in enumerate(bc_average_window_sizes):
            bc_average_features[:, i] = purchase_history[:, -w:].mean(axis=1)

        # average window buycounts
        bc_average_features_category = np.zeros((len(y), len(bc_average_window_sizes)))
        _purchase_history_category = tmp[2].dot(aggregation_j_c)
        _purchase_history_category_by_j = _purchase_history_category[:, :, data_j.c]
        _purchase_history_category_by_j = _purchase_history_category_by_j.swapaxes(
            1, 2
        ).reshape(-1, T)
        for i, w in enumerate(bc_average_window_sizes):
            bc_average_features_category[:, i] = _purchase_history_category_by_j[
                :, -w:
            ].mean(axis=1)

        # compile features
        x = np.hstack(
            [
                np.ones_like(y)[:, np.newaxis],  # B x 1
                frequencies[:, np.newaxis],  # B x 1
                cosine_similarity[:, np.newaxis],  # B x 1
                discounts_cross,  # B x J
                bc_average_features,  # B x W
                discounts_cross.dot(aggregation_j_c),  # B x C
                bc_average_features_category,  # B x W (category)
            ]
        )

        n_samples += len(y)
        needed_samples = N - n_samples

        if needed_samples < 0:
            y = y[:needed_samples]
            x = x[:needed_samples, :]

        # store
        y_list.append(y)
        x_list.append(x)

    logger.info("done")

    index_it_used = pd.concat(idx_list).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        index_it_used,
        index_it.iloc[range(index_it_used.shape[0])],
    )

    return np.concatenate(y_list), np.concatenate(x_list), index_it_used
