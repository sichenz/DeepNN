# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import torch
import pickle
import scipy.sparse
import modules.lib
import numpy as np
import pandas as pd
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler

from loguru import logger


def to_torch_sparse(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


class DatasetDNN(Dataset):
    def __init__(self, _y, _np, _bc, _pf):
        self._y = _y
        self._np = _np
        self._bc = _bc
        self._pf = _pf

    @classmethod
    def from_streamer(cls, file_sparse_features, streamer, batch_size=4096, I=None):
        if os.path.isfile(file_sparse_features) and (I is None):
            raise NotImplementedError()
            # logger.info("load sparse features")
            # _y, _np, _bc, _pf = modules.lib.load_pickle(file_sparse_features)
        else:
            logger.info("create sparse features")
            streamer.reset_streamer(randomize=True)
            if batch_size > streamer.num_training_samples:
                logger.info("reduce batch_size to streamer.num_training_samples")
                batch_size = streamer.num_training_samples
            ys = []
            nps = []
            bcs = []
            pfs = []
            while True:
                try:
                    data_batch = streamer.get_batch(batch_size)
                    ys.append(scipy.sparse.coo_matrix(data_batch[0]))
                    nps.append(scipy.sparse.coo_matrix(data_batch[1]))
                    bcs.append(
                        scipy.sparse.coo_matrix(data_batch[2].reshape((batch_size, -1)))
                    )
                    pfs.append(scipy.sparse.coo_matrix(data_batch[3]))
                except IndexError:
                    break
            _y = to_torch_sparse(scipy.sparse.vstack(ys))
            _np = to_torch_sparse(scipy.sparse.vstack(nps))
            _bc = to_torch_sparse(scipy.sparse.vstack(bcs))
            _pf = to_torch_sparse(scipy.sparse.vstack(pfs))
            # modules.lib.dump_pickle(x=(_y, _np, _bc, _pf), f=file_sparse_features)
        logger.info("create dense features")
        _y = _y.to_dense()
        _np = _np.to_dense()
        _pf = _pf.to_dense()
        _bc = _bc.to_dense()
        return cls(_y, _np, _bc, _pf)

    def __getitem__(self, ind: torch.LongTensor):
        return self._y[ind, :], self._np[ind, :], self._bc[ind, :], self._pf[ind, :]

    def __len__(self):
        return self._y.shape[0]


class BatchSamplerDNN(BatchSampler):
    def __init__(self, source, batch_size, shuffle=False, drop_last=False):
        super(BatchSamplerDNN, self).__init__(
            sampler=RandomSampler(data_source=source),
            batch_size=batch_size,
            drop_last=drop_last,
        )
        self.n = len(source)
        self.shuffle = shuffle

    def __iter__(self) -> torch.LongTensor:
        if self.shuffle:
            indices = torch.randperm(self.n)
        else:
            indices = torch.arange(self.n)
        index = 0
        while index < self.n:
            yield indices[index : index + self.batch_size]
            index += self.batch_size

    def __len__(self) -> int:
        if self.drop_last:
            return self.n // self.batch_size
        else:
            return (self.n + self.batch_size - 1) // self.batch_size


class DataLoaderDNN(DataLoader):
    @property
    def _auto_collation(self):
        return False

    @property
    def _index_sampler(self):
        return self.batch_sampler


def get_data_loader(c, l, p, ptmp, prod2id, I, do_pickle=False):
    """
    c=config_train_streamer
    l="train"
    p=path_data
    ptmp=re.sub(os.getenv("HOME"), '/mnt', path_data)
    prod2id=prod2id
    I=I_train
    do_pickle=pickle
    """

    logger.info(f"path_data={p}")
    logger.info(f"path_tmp={ptmp}")

    md5_config_streamer = modules.lib.md5_dict(c)
    os.makedirs(p, exist_ok=True)
    file_dataset = f"{ptmp}/tmp/{l}_{md5_config_streamer}.pickle"
    # logic to decide whether to load or create data loader
    if do_pickle:
        do_load = os.path.isfile(file_dataset)
        if not do_load:
            logger.info(
                f"no {l} data_loader available for this config, create new data_loader"
            )
    else:
        do_load = False
        logger.info(f"force creation of {l} data_loader")

    # load or create data loader
    if do_load:
        logger.info(f"load {l} data_loader ({file_dataset})")
        # with open(file_dataset, "rb") as con:
        #    dataset_dnn = pickle.load(con)
        logger.info(f"finished loading data_loader")
    else:
        logger.info("load basket and action data")
        baskets = pd.read_parquet(f"{p}/baskets.parquet")
        baskets["quantity"] = 1.0
        file_actions = f"{p}/action.parquet"
        if os.path.isfile(file_actions):
            actions = pd.read_parquet(file_actions)
        else:
            actions = pd.DataFrame({"i": [0], "j": 0, "t": 88, "discount": 0})
        if I is not None:
            logger.info(f"reduce number of consumers to {I:,}")
            baskets = baskets[baskets["i"] < I]
            actions = actions[actions["i"] < I]
        logger.info(f"create data_streamer")
        streamer = modules.data_streamer_v1.ProductChoiceDataStreamer(
            basket_data=baskets,
            action_data=actions,
            prod2id=prod2id,
            **c,
        )
        # os.makedirs(f"{ptmp}/tmp", exist_ok=True)
        logger.info(f"create data_loader")
        dataset_dnn = modules.data_loader.DatasetDNN.from_streamer(
            file_sparse_features=f"{ptmp}/tmp/{md5_config_streamer}.pkl",
            streamer=streamer,
            batch_size=min(streamer.num_training_samples, 4096),
            I=I,
        )
        if do_pickle:
            logger.info(f"save data_loader")
            # with open(file_dataset, "wb") as con:
            #    pickle.dump(dataset_dnn, con, protocol=4)
        logger.info(f"finished creating data_loader")
    return dataset_dnn
