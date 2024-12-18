# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import sys
import os
import copy
import collections
import re
import glob
import scipy.stats

import pandas as pd
import numpy as np

import sklearn, sklearn.preprocessing

import matplotlib.pyplot as plt

import joblib
from joblib import Parallel, delayed

import warnings

warnings.filterwarnings(
    "ignore",
    message="A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.",
)


# baselines
class Agent:
    def __init__(self, **kwargs):

        self.name = "NoneAgent"
        self.campaigns = kwargs.get("campaigns", None)
        self.discounts = kwargs.get("discounts", [0.1, 0.2, 0.3, 0.4])
        self.verbose = kwargs.get("verbose", 0)

    def feed(self, *args):
        return None


class RandomAgent(Agent):
    """
    agent assigns coupons randomly
    """

    def __init__(self, **kwargs):

        Agent.__init__(self, **kwargs)
        self.gym = kwargs.get("gym", None)
        if self.gym is not None:
            self.I = self.gym.I
        else:
            self.I = kwargs.get("I", 501)
        self.name = "RandomAgent"

    def feed(self, *args):
        action = pd.DataFrame(
            {
                "i": list(range(self.I)),
                "j": np.random.choice(self.campaigns, self.I, replace=True),
                "discount": np.random.choice(self.discounts, self.I, replace=True),
                "delta_er": np.nan,
            }
        )
        return action


class RandomAgentN:
    def __init__(self, data_j, discounts, n_coupons, I, seed=501, **kwargs):

        self.name = "RandomAgentN"
        self.data_j = data_j
        self.discounts = discounts
        self.N = n_coupons
        self.I = I
        self.seed = seed
        self.categories = self.data_j.c.unique()
        self.C = len(self.categories)
        if self.N > self.C:
            raise ValueError("n_coupons may not be larger than number of categories.")
        _tmp_n_j_by_c = self.data_j.groupby("c").j.count().values
        assert len(np.unique(_tmp_n_j_by_c)) == 1
        self.n_j_by_c = _tmp_n_j_by_c[0]

    def feed(self, seed_offset=0):
        np.random.seed(501 + seed_offset)
        _tmp = np.random.uniform(size=(self.I, self.C))
        coupons = pd.DataFrame(
            {
                "i": np.repeat(range(self.I), self.N),
                "c": self.categories[_tmp.argsort()[:, : self.N].flatten()],
            }
        )
        coupons["offset"] = np.random.choice(self.n_j_by_c, coupons.shape[0])
        coupons["j"] = coupons["c"] * self.n_j_by_c + coupons["offset"]
        coupons["discount"] = np.random.choice(self.discounts, coupons.shape[0])
        coupons.drop(columns=["c", "offset"], inplace=True)
        coupons["nc"] = coupons.groupby(["i"]).cumcount()
        return coupons
