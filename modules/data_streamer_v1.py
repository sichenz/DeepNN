# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import collections
import itertools
import scipy.sparse

import numpy as np


class ProductChoiceDataStreamer:
    """
    Stream Data for a DNN Product Choice Models (shared streamer between architectures).

    Currently we assume that t is an integer time index.

    The timeline for the training streamer looks like this:
        0                                                                                             T
        |---------------------------------------------------------------------------------------------|
        |     | <-- history --> | <---------------- training_data -------------> | <- testing_data -> |
        |     |-----------------|*-----------------------------------------------*--------------------|
        |                       |time_first                             time_last|
    shifted time index
              0
        |-----| ...
    """

    def __init__(
        self,
        basket_data,
        action_data,
        prod2id,
        cust2id=None,
        id2cust=None,
        time_first=10,
        time_last=90,
        history_length=10,
        randomize=True,
        full_history_pf=True,
        debug=False,
    ):

        assert time_first >= history_length

        # store input variables
        self.time_first = time_first
        self.time_last = time_last
        self.history_length = history_length
        self.time_offset = self.time_first - self.history_length
        self.full_history_pf = full_history_pf

        # product-id map (use input because this should align with map in pretraining of
        # product embedding)
        self.prod2id = prod2id
        self.id2prod = {self.prod2id[k]: k for k in self.prod2id}
        self.J = len(self.prod2id)

        # user-id map (prune data to relevant weeks)
        if cust2id is not None and id2cust is not None:
            self.cust2id = cust2id
            self.id2cust = id2cust
        else:
            self.cust2id, self.id2cust = self._init_cust_id(basket_data)
        self.I = len(self.cust2id)

        # preprocess data
        basket_data = self._preprocess_data(basket_data)
        action_data = self._preprocess_data(action_data)

        # user-time pairs = training samples
        #   depends on number of users (after pruning) and number of weeks
        self.user_time_pairs = self._get_user_time_pairs()
        self.num_training_samples = len(self.user_time_pairs)
        self.randomize = randomize
        self.reset_streamer(randomize=self.randomize)

        # turn data into sparse matrices
        #   allows fast subsetting in streaming
        self.basket_csr = self._init_sparse_matrix(
            x=basket_data, value_variable="quantity"
        )
        self.action_csr = self._init_sparse_matrix(
            x=action_data, value_variable="discount"
        )

        basket_data_cut = basket_data[basket_data["t"] < self.time_offset].reset_index(
            drop=True
        )
        basket_data_cut["t"] = self.time_first - self.history_length
        self.buycounts_cut_csr = self._init_sparse_matrix(
            x=basket_data_cut, value_variable="quantity"
        )[:, : self.J]

        self.debug = debug

    def get_batch(self, batch_size):
        curr_baskets = [
            self.user_time_pairs_cache.popleft() for _i in range(batch_size)
        ]
        i_idx, t_idx = zip(*curr_baskets)

        i_idx = np.array(i_idx)
        t_idx = np.array(t_idx)
        t_idx_shifted = t_idx - self.time_offset
        B = len(i_idx)
        Tb = max(t_idx_shifted) + 1
        basket_batch = self.basket_csr[i_idx, : (Tb * self.J)].toarray()
        basket_batch = basket_batch.reshape((B, Tb, self.J))

        batch_labels = basket_batch[range(B), t_idx_shifted, :]

        t_idx_arr_rev = Tb - t_idx_shifted
        idx_0_dim_b = np.repeat(range(B), t_idx_arr_rev)
        idx_0_dim_t = (
            Tb
            - (
                np.repeat(t_idx_arr_rev - t_idx_arr_rev.cumsum(), t_idx_arr_rev)
                + np.arange(t_idx_arr_rev.sum())
            )
            - 1
        )
        basket_batch[idx_0_dim_b, idx_0_dim_t, :] = 0

        if self.full_history_pf:
            batch_freq = (
                np.sum(basket_batch, axis=1)
                + self.buycounts_cut_csr[i_idx, :].toarray()
            ) / t_idx[:, np.newaxis]
        else:
            batch_freq = np.sum(basket_batch, axis=1) / t_idx_shifted[:, np.newaxis]

        idx_t_window = (
            t_idx_shifted.reshape(B, 1)
            + np.arange(self.history_length).reshape(1, self.history_length)
            - self.history_length
        )
        batch_histories = basket_batch[np.arange(B).reshape((B, 1)), idx_t_window, :]

        batch_actions = self.action_csr[
            i_idx.reshape(B, 1),
            t_idx_shifted.reshape(B, 1) * self.J
            + np.arange(self.J).reshape((1, self.J)),
        ].toarray()

        if self.debug:
            return (
                batch_labels,
                batch_actions,
                batch_histories,
                batch_freq,
                i_idx,
                t_idx,
            )
        return batch_labels, batch_actions, batch_histories, batch_freq, (i_idx, t_idx)

    def reset_streamer(self, randomize=True):
        self.shuffled_user_time_pairs = list(self.user_time_pairs)
        if randomize:
            np.random.shuffle(self.shuffled_user_time_pairs)
        self.user_time_pairs_cache = collections.deque(self.shuffled_user_time_pairs)

    def _init_cust_id(self, basket_data):
        basket_data_pruned = basket_data[
            (basket_data["t"] >= self.time_first) & (basket_data["t"] <= self.time_last)
        ]
        all_customer = basket_data_pruned.i.unique()
        cust2id = dict()
        id2cust = dict()
        for i in range(len(all_customer)):
            cust2id[all_customer[i]] = i
            id2cust[i] = all_customer[i]
        return cust2id, id2cust

    def _preprocess_data(self, x):
        x["j"] = x["j"].map(self.prod2id)
        x = x[x.i.isin(self.id2cust.values())].reset_index(drop=True)
        x["i"] = x["i"].map(self.cust2id)
        return x

    def _get_user_time_pairs(self):
        t_set = list(range(self.time_first, self.time_last + 1))
        i_set = list(range(self.I))
        return list(itertools.product(i_set, t_set))

    def _init_sparse_matrix(self, x, value_variable):
        x_subset = x[
            (x["t"] >= (self.time_first - self.history_length))
            & (x["t"] <= self.time_last)
        ].reset_index(drop=True)
        x_subset["t"] -= self.time_first - self.history_length
        rows = x_subset["i"].values
        columns = (x_subset["j"] + self.J * x_subset["t"]).values
        data = x_subset[value_variable].astype(float).values
        shape = (
            self.I,
            self.J * (self.history_length + (self.time_last - self.time_first) + 1),
        )
        return scipy.sparse.csr_matrix((data, (rows, columns)), shape=shape)
