# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import numpy as np
import pandas as pd

# from loguru import logger


def build_dcm_dataset(
    streamer,
    jx,
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
    """

    if False:
        streamer = streamer_train
        jx = 0
        batch_size = 5_000
        w2v_embedding = w2v_embedding_norm
        bc_average_window_sizes = config["training"]["avg_windows"]
        randomize = True

    # logger.info("build dcm dataset cross by j")

    streamer.reset_streamer(randomize=randomize)
    J = streamer.J
    T = streamer.history_length

    y_list = []
    x_list = []
    idx_list = []

    index_it = pd.DataFrame(streamer.shuffled_user_time_pairs, columns=["i", "t"])

    sampler_empty = False
    cnt = 0
    while not sampler_empty:

        batch_size_ix = min([len(streamer.user_time_pairs_cache), batch_size])
        if batch_size_ix == 0:
            break
        tmp = streamer.get_batch(batch_size_ix)
        idx_list.append(
            pd.DataFrame(
                {
                    "i": tmp[4][0],
                    "t": tmp[4][1],
                }
            )
        )

        # build y
        y = tmp[0][:, jx]

        # build x
        discounts = tmp[1][:, jx]
        discounts_cross = tmp[1]
        frequencies = tmp[3][:, jx]

        # cs
        purchase_history_jx = tmp[2][:, :, jx]
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
        cosine_similarity = np.dot(user_vector_norm, w2v_embedding.T)[:, jx]

        # average window buycounts
        bc_average_features = np.zeros((len(y), len(bc_average_window_sizes)))
        _buycounts = tmp[2][:, :, jx]

        for i, w in enumerate(bc_average_window_sizes):
            bc_average_features[:, i] = _buycounts[:, -w:].mean(axis=1)

        # compile features
        x = np.hstack(
            [
                np.ones_like(discounts)[:, np.newaxis],  # B x 1
                frequencies[:, np.newaxis],  # B x 1
                cosine_similarity[:, np.newaxis],  # B x 1
                discounts_cross,  # B x J
                bc_average_features,  # B x W
            ]
        )

        # store
        y_list.append(y)
        x_list.append(x)

    # logger.info("done")

    index_it_used = pd.concat(idx_list).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        index_it_used,
        index_it.iloc[range(index_it_used.shape[0])],
    )

    return np.concatenate(y_list), np.concatenate(x_list), index_it_used