# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import copy
import gensim.models
import sklearn.manifold
import scipy.special
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from loguru import logger

import modules.args
import modules.lib


# data
def baskets_df_to_list(x, min_basket_size=2, shuffle=True, to_string=True, seed=501):

    # create raw basket list
    x_basket_product = x[["basket_hash", "j"]]
    keys, values = x_basket_product.sort_values("basket_hash").values.T
    ukeys, index = np.unique(keys, True)
    arrays = np.split(values, index)
    basket_list = [list(set(a)) for a in arrays[1:]]

    # format basket list
    basket_list_out = []
    for basket in basket_list:
        if len(basket) >= min_basket_size:
            if to_string:
                basket_list_out.append([str(x) for x in basket])
            else:
                basket_list_out.append(basket)

    # randomise basket order and product order in baskets
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(basket_list_out)
        for i in range(len(basket_list_out)):
            np.random.shuffle(basket_list_out[i])

    return basket_list_out


def gensim_embedding_to_pandas(w2v, J=250):
    """
    extract embeddings and vocabulary from gensim word2vec model
    """
    vocab = []
    for key, value in w2v.wv.vocab.items():
        vocab.append(
            pd.DataFrame({"j": [np.int64(key)], "index": value.index, "N": value.count})
        )
    vocab = pd.concat(vocab).set_index("index").sort_index()

    syn0 = pd.DataFrame(w2v.wv.vectors, index=vocab["j"]).sort_index()

    # add missing products
    missing_products = set(range(J)).difference(set(syn0.reset_index().j.values))
    for j in missing_products:
        logger.info(f"add missing product vector for product {j}")
        syn0.loc[j] = syn0.mean()
    syn0 = syn0.sort_index()

    if w2v.negative > 0:
        syn1 = pd.DataFrame(w2v.trainables.syn1neg, index=vocab["j"]).sort_index()
    else:
        syn1 = pd.DataFrame(w2v.trainables.syn1, index=vocab["j"]).sort_index()

    return syn0, syn1, vocab


# callbacks
class CallbackSaveWeights(gensim.models.callbacks.CallbackAny2Vec):
    """
    Callback to save gensim weights on epoch end.
    """

    def __init__(self, path=None):
        self.syn0 = []
        self.syn1 = []
        super(CallbackSaveWeights, self).__init__()
        self.path = path
        self.counter_epoch = 0

    def on_epoch_end(self, model):
        # get data
        syn0_epoch = copy.deepcopy(model.wv.vectors)
        if model.negative > 0:
            syn1_epoch = copy.deepcopy(model.trainables.syn1neg)
        else:
            syn1_epoch = copy.deepcopy(model.trainables.syn1)

        # add weights to class list
        self.syn0.append(syn0_epoch)
        self.syn1.append(syn1_epoch)

        # save data to disk
        if self.path is not None:
            np.save(
                os.path.join(self.path, "gensim_syn0_%04d.npy" % self.counter_epoch),
                syn0_epoch,
            )
            np.save(
                os.path.join(self.path, "gensim_syn1_%04d.npy" % self.counter_epoch),
                syn1_epoch,
            )
        self.counter_epoch += 1


class CallbackDeltaWV(gensim.models.callbacks.CallbackAny2Vec):
    """
    Callback to compute the aggregate change in embedding weights
    between epochs.
    """

    def __init__(self, out=None):
        self.deltawv = []
        self.wv_last = None
        self.out = out
        super(CallbackDeltaWV, self).__init__()

    def on_epoch_end(self, model):

        if self.wv_last is not None:
            self.deltawv.append(((model.wv.vectors - self.wv_last) ** 2).sum())

        self.wv_last = copy.deepcopy(model.wv.vectors)

    def on_train_end(self, model):
        if self.out:
            np.savetxt(self.out, self.deltawv)


class CallbackHoldoutSet(gensim.models.callbacks.CallbackAny2Vec):
    """
    callback to compute loss on validation holdout set. Loss can be
    either negative sampling loss (negative=1, in principle faster) or
    the full softmax loss (negative=0, in principle slower). Samples
    can be either skip-gram (sg>0) or cbow (sg=0) style. The class needs
    to be initialized with a list of basket data. The class will generate
    the corresponding validation samples.
    """

    def __init__(
        self, holdout, negative=0, sg=0, random=0, log_nth_batch=100, verbose=0
    ):

        self.holdout = holdout
        self.context = self.center = None
        self.negative = negative  # number of negative samples
        self.sg = sg
        self.random = random
        self.losses = []
        self.losses_batch = []
        self.log_nth_batch = log_nth_batch
        self.epoch = 0
        self.epoch_batch = 0
        self.verbose = verbose
        super(CallbackHoldoutSet, self).__init__()

    def _log(self, x):
        if self.verbose > 0:
            if self.verbose > 1 or isinstance(x, str):
                logger.info(x)

    def _setup_sg_pairs(self, model):
        #
        # Construct skip-gram style center-context pairs.
        #
        # Here, the "center" word appears on the input layer and is used to predict the
        # context. All pairs of words within a certain window size of each other appear
        # as training samples.
        #

        # map tokens to index
        token_map = {k: v.index for k, v in model.wv.vocab.items()}
        all_token_indices = set([model.wv.vocab[k].index for k in model.wv.vocab])

        # loop through all sentences
        _tmp_context = []
        _tmp_center = []
        for sentence in self.holdout:
            # build all center-context pairs
            sentence = [token_map[k] for k in sentence if k in token_map]
            samples_sentence = pd.DataFrame(
                {
                    "w1": np.repeat(sentence, len(sentence)),
                    "pos1": np.repeat(np.arange(len(sentence)), len(sentence)),
                    "w2": np.tile(sentence, len(sentence)),
                    "pos2": np.tile(np.arange(len(sentence)), len(sentence)),
                }
            )

            # limit to center-context pairs within window `w`
            samples_sentence["delta_pos"] = np.abs(
                samples_sentence["pos1"] - samples_sentence["pos2"]
            )
            samples_sentence = samples_sentence[
                (samples_sentence["delta_pos"] != 0)
                & (samples_sentence["delta_pos"] <= model.window)
            ]

            # generate negative samples
            allowed_negative_samples = list(all_token_indices.difference(set(sentence)))
            negative_samples = np.random.choice(
                allowed_negative_samples, (samples_sentence.shape[0], model.negative)
            )

            # format output
            _tmp_context.append(
                samples_sentence["w1"].values.reshape(-1, 1).astype(np.int32)
            )
            _tmp_center.append(
                np.hstack(
                    [
                        samples_sentence["w2"].values.reshape(-1, 1).astype(np.int32),
                        negative_samples,
                    ]
                )
            )

        self.context = np.concatenate(_tmp_context)
        self.center = np.concatenate(_tmp_center)

    def _calculate_loss(self, model):

        # first time through, compute the training samples from sentences
        # can't do this on init, because we need model first
        if self.context is None:
            self._log("build context center pairs")
            if self.random:
                raise NotImplementedError
            elif self.sg:
                self._setup_sg_pairs(model)
            else:
                self._setup_cbow_pairs(model)
            self.n_samples = len(self.center) * (
                self.negative + 1
            )  # to get loss per sample

        loss = self._calculate_loss_vectorized(model)

        return loss

    def _calculate_loss_vectorized(self, model):

        # Efficiently compute loss using vectorized numpy operations.
        l1 = (
            np.einsum("ijk->ik", model.wv.vectors[self.context]) / self.context.shape[1]
        )
        if self.center.shape[1] == 1:
            # softmax
            raise NotImplementedError
        else:  # negative sampling
            l2 = model.trainables.syn1neg[self.center]
            scores = -np.einsum("ik,ijk->ij", l1, l2)
            scores[:, 0] *= -1
            scores = np.float64(scores)
            loss = -np.sum(np.log(scipy.special.expit(scores)))

        return loss

    def on_batch_end(self, model):
        if self.epoch_batch != 0 and self.epoch_batch % self.log_nth_batch == 0:
            self._log("batch callback -- %d" % self.epoch_batch)
            batch_end_loss = self._calculate_loss(model)
            self.losses_batch.append(batch_end_loss)
        self.epoch_batch += 1
        return

    def on_epoch_begin(self, model):
        if self.epoch == 0:
            self.losses.append(self._calculate_loss(model))
        self._log("epoch callback -- %d" % self.epoch)
        self.epoch_batch = 0
        return

    def on_epoch_end(self, model):
        epoch_end_loss = self._calculate_loss(model)
        self.losses.append(epoch_end_loss)
        self.epoch += 1
        return


# plots
def plot_loss(
    test=None,
    validation=None,
    train=None,
    normalise=True,
    batch=False,
    remove_first_value=False,
    figsize=(10, 7),
):
    """plot loss at end of epoch

    test, validation, train: loss callbacks
    normalise (True): loss per sample?
    batch (False): plot batch loss? otherwise plot epoch loss
    remove_first_value (False): remove first value in loss array (loss before training)
    figsize: figure size
    """

    def plot_loss_single(x, label, normalise, batch, remove_first_value):
        if x is not None:
            losses = np.array(x.losses_batch) if batch else np.array(x.losses)
            if remove_first_value:
                losses = losses[1:]
            if normalise:
                losses /= x.n_samples
            _ = plt.plot(losses, label=label)

    p = plt.figure(figsize=figsize)
    plot_loss_single(test, "test", normalise, batch, remove_first_value)
    plot_loss_single(validation, "validation", normalise, batch, remove_first_value)
    plot_loss_single(train, "train", normalise, batch, remove_first_value)
    _ = plt.legend()
    _ = plt.xlabel("batch" if batch else "epoch")
    _ = plt.ylabel("loss (per sample)" if normalise else "loss")
    return p


def plot_dW(x, normalise=True, logy=False, figsize=(10, 7)):
    """plot weight change for product embedding

    x: gensim callback
    normalise (True): dw per sample?
    logy (False): log scale for y axis?
    """
    p = plt.figure(figsize=figsize)
    dw_values = x.deltawv
    if normalise:
        dw_values /= np.prod(x.wv_last.shape)
    _ = plt.plot(dw_values)
    if logy:
        _ = plt.semilogy()
    _ = plt.xlabel("epoch")
    _ = plt.ylabel("dW (per weight)" if normalise else "sum(dW)")
    return p


def main(x, path_data, seed_offset, **kwargs):

    # CONFIG
    config = modules.lib.read_yaml(x)
    config["path_data"] = path_data
    logger.info(f"path_data={path_data}")
    dir_model = "gensim"

    # create output path
    path_gensim_results = f"{path_data}/{dir_model}/results"
    os.makedirs(path_gensim_results, exist_ok=True)

    # check state
    # check whether results exists
    file_gym0 = f"{path_data}/gym0_light.pickle.gz"
    file_result = f"{path_data}/{dir_model}/model.w2v"
    if modules.lib.check_state(file_gym0, 10, file_result, path_data):
        return 0
    modules.lib.touch(file_result)

    # log experiment
    experiment_tracker = {
        "w2v_parameters": config["w2v"],
        "dir": dir_model,
        "status": "started",
        "split": {"n_validation": 20000, "n_test": 20000},
    }
    modules.lib.write_yaml(experiment_tracker, f"{path_gensim_results}/experiment.yaml")

    # load data
    baskets = pd.read_parquet(f"{path_data}/baskets.parquet")
    baskets["basket_hash"] = baskets.eval("1000000000 + t*1000000 + i")
    product = pd.read_csv(f"{path_data}/data_j.csv", usecols=["j", "c"])

    # build basket list
    baskets_list = baskets_df_to_list(
        x=baskets, min_basket_size=2, shuffle=True, to_string=True
    )

    config_split = experiment_tracker["split"]
    test = baskets_list[: config_split["n_test"]]
    train = baskets_list[config_split["n_test"] :]
    validation = baskets_list[
        config_split["n_test"] : (config_split["n_test"] + config_split["n_validation"])
    ]
    logger.info(f"n_train_samples={len(train):,}")
    logger.info(f"n_validation_samples={len(validation):,}")
    logger.info(f"n_test_samples={len(test):,}")

    # train product embedding
    gensim_callback = [
        # [0] = save weights
        CallbackSaveWeights(path=path_gensim_results),
        # [1] = changes in weight matrix
        CallbackDeltaWV(),
        # [2] = test loss
        CallbackHoldoutSet(
            test,
            sg=experiment_tracker["w2v_parameters"]["sg"],
            negative=experiment_tracker["w2v_parameters"]["negative"],
            log_nth_batch=100_000_000,
        ),
        # [3] = validation loss
        CallbackHoldoutSet(
            validation,
            sg=experiment_tracker["w2v_parameters"]["sg"],
            negative=experiment_tracker["w2v_parameters"]["negative"],
            log_nth_batch=100_000_000,
            verbose=1,
        ),
    ]

    w2v = gensim.models.Word2Vec(
        train,
        callbacks=gensim_callback,
        workers=8,
        seed=501,
        **experiment_tracker["w2v_parameters"],
    )

    # dW and loss
    logger.info("plot dW and loss")
    p_dW = plot_dW(gensim_callback[1], figsize=(18, 6))
    p_dW.savefig(f"{path_data}/{dir_model}/plot_dW.png")
    p_loss = plot_loss(
        validation=gensim_callback[3],
        test=gensim_callback[2],
        normalise=True,
        batch=False,
        remove_first_value=True,
        figsize=(18, 6),
    )
    p_loss.savefig(f"{path_data}/{dir_model}/plot_loss.png")

    # heatmap
    logger.info("plot heatmap")
    w2v_embedding, _, w2v_vocabulary = gensim_embedding_to_pandas(w2v)
    w2v_embedding = (
        w2v_embedding.merge(product, on="j", how="left")
        .set_index(["c", "j"])
        .sort_index()
    )
    p_heatmap = plt.figure(figsize=(18, 6))
    _ = sns.heatmap(w2v_embedding.values.T)
    p_heatmap.savefig(f"{path_data}/{dir_model}/plot_heatmap.png")

    # close all figures
    plt.close("all")

    # embedding for downstream applications
    w2v_embedding_j = w2v_embedding.reset_index()
    w2v_embedding_j.columns = [str(x) for x in w2v_embedding_j.columns]

    # save results
    logger.info("save results")
    w2v.callbacks = None
    w2v.save(f"{path_data}/{dir_model}/model.w2v")
    w2v_vocabulary.to_csv(f"{path_data}/{dir_model}/w2v_vocabulary.csv", index=False)
    w2v_embedding.to_csv(f"{path_data}/{dir_model}/w2v_embedding.csv", index=False)
    w2v_embedding_j.to_parquet(f"{path_data}/w2v_embedding_j.parquet")

    # update experiment tracker
    experiment_tracker["status"] = "done"
    modules.lib.write_yaml(experiment_tracker, f"{path_gensim_results}/experiment.yaml")

    logger.info("□")


if __name__ == "__main__":

    args = modules.args.global_args("Train product2vec.")
    all_path_data = modules.lib.get_data_paths(args)

    for name, params in all_path_data.items():
        if modules.lib.check_bootstrap_iter(params["seed_offset"], args.y, args.z):
            continue
        main(x=args.c, **params)
