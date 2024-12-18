# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import os
import pickle
import numpy as np
import pandas as pd
import scipy.stats

from loguru import logger

from pandas.testing import assert_frame_equal

import warnings

np.set_printoptions(linewidth=120)


class SupermarketGym:
    """
    A) DATA_I
        1) customer consumption heterogeneity `w_i_cons`
            EITHER  column in data_i
            OR      w_i_cons_min + w_i_cons_max + delta_W_cons
        2) latent consumer tastes `W_i`
            EITHER  np.array, shape = (n_hidden, I)
            OR      n_hidden
        3) data_beta_i_p
            EITHER  pd.DataFrame with columns i and beta_i_p
            OR      mu_ps + sigma_ps
            OR      price_sensitivity_levels

    B) DATA_C
        1) category consumption rate `cons_c`
            EITHER  column in data_c
            OR      cons_c_min + cons_c_max
        2) base utility for purchasing in category c `gamma_c`
            EITHER  column in data_c
            OR      gamma_c_min + gamma_c_max
        3) base price product products in category c `p_c`
            EITHER  column in data_c
            OR      mu_p and sigma_p
        4) sensitivity for purchasing in category c given current inventory
            EITHER  column in data_c
            OR      gamma_c_inv_mu + gamma_c_inv_sigma

    C) DATA_IC
        1) inventory at t=0
            EITHER  _inventory_0
            OR      lambda_0_inv
        2) user-category consumption `cons_ic`
            EITHER  cons_ic
            OR      delta_cons
        3) latent category vectors `Gamma_c`
            EITHER  np.array, shape = (C, n_hidden)
            OR      n_hidden + scale_Gamma_c
        4) user-category base preferences `gamma_ic`
            EITHER  pd.DataFrame with columns i, c, gamma_ic
            OR      input for W_i + Gamma_c
        5) category own-price effect `gamma_ic_p`
            EITHER  pd.DataFrame with columns i, c, gamma_ic_p
            OR      mu_gamma_p + sigma_gamma_p
        6) category cross-price effect `gamma_ick_cp_total`
            delta_cp + sigma_cp + prob_cp

    D) DATA_IJ
        1) mean of latent product vectors `mu_Beta_jc`
            EITHER  np.array, shape = (n_hidden)
            OR      mu_Beta_jc_min + mu_Beta_jc_max + n_hidden
        2) latent product vectors `Beta_jc`
            EITHER  np.array, shape = (J, n_hidden)
            OR      scale_Beta_jc + n_hidden + input for mu_Beta_jc
        3) user-product base preferences `beta_ijc`
            EITHER  pd.DataFrame with columns i, j, beta_ijc
            OR      input for W_i + Beta_jc
    """

    def __init__(
        self,
        ## GENERAL SETUP
        I=None,  # number of consumers
        data_c=None,  # category master (c, cons_c, [p_c])
        data_j=None,  # product master (j, c, [p_jc, beta_j])
        data_i=None,  # user master (i, [])
        burnin_action=None,  # coupons to be applied during burnin
        # A1)
        delta_W_cons=None,  # user-specific category consumption
        w_i_cons_min=None,  #
        w_i_cons_max=None,  #
        # A2)
        W_i=None,
        n_hidden=None,
        # A3)
        data_beta_i_p=None,
        mu_ps=None,
        sigma_ps=None,
        price_sensitivity_levels=None,
        # B1)
        cons_c_min=None,
        cons_c_max=None,
        # B2)
        gamma_c_min=None,
        gamma_c_max=None,
        # B3)
        mu_p=None,
        sigma_p=None,
        # B4)
        gamma_c_inv_mu=None,
        gamma_c_inv_sigma=None,
        # C1) starting inventory
        _inventory_0=None,
        lambda_0_inv=None,
        # C2) consumption
        cons_ic=None,
        delta_cons=None,
        # C3) base
        Gamma_c=None,
        scale_Gamma_c=None,
        # C4) base
        gamma_ic=None,
        # C5) own-category price
        gamma_ic_p=None,
        mu_gamma_p=None,
        sigma_gamma_p=None,
        # C6) cross-category price
        cp_type=None,
        delta_cp=None,
        min_mu_cp=None,
        max_mu_cp=None,
        sigma_cp=None,
        prob_cp=None,
        # D1)
        mu_Beta_jc=None,
        mu_Beta_jc_min=None,
        mu_Beta_jc_max=None,
        # D2)
        Beta_jc=None,
        scale_Beta_jc=None,
        # D3)
        beta_ijc=None,
        ## CONTROL
        burn_in=50,  # number of weeks for burn-in
        do_burn_in=True,
        own_price_method="probability",
        seed=501,  # seed for random number generator
        save_data_burnin=True,
        verbose=1,
    ):

        ## CONTROL
        self.seed = seed
        np.random.seed(self.seed)  # set seed to ensure reproducibility
        self.save_data_burnin = save_data_burnin
        self.verbose = verbose

        ## process user-specified master tables

        # ... product master (`data_j`)
        assert isinstance(data_j, pd.DataFrame)
        assert all([i in data_j.columns for i in ["j", "c"]])
        self.J = data_j.j.nunique()
        self.C = data_j.c.nunique()

        # ... user master (`data_i`)
        if isinstance(data_i, pd.DataFrame):
            if I is None:
                I = data_i.shape[0]
            else:
                assert I == data_i.shape[0]

        ## SETUP
        self.I = I
        self.burn_in = burn_in
        self.do_burn_in = do_burn_in
        self.week = -burn_in

        # number of latent consumer tastes
        self.n_hidden = n_hidden

        # weights for average price effects over products for each user-category
        # combination
        self.own_price_method = own_price_method

        ## logging
        self._log(f"I = {self.I:,}")
        self._log(f"J = {self.J:,}")
        self._log(f"C = {self.C:,}")
        self._log(f"n_weeks_burnin = {self.burn_in}")
        if burnin_action is None:
            self._log("no coupons provided for burn in")
        else:
            self._log(f"n_coupons_burnin = {burnin_action.shape[0]:,}")
        if self.save_data_burnin:
            self._log("keeping burnin data")

        ## I ##

        # data_i = user-level data
        if data_i is None:
            self._log("build data_i")
            self.data_i = pd.DataFrame(
                {
                    "i": list(range(self.I)),
                }
            )
        else:
            self.data_i = data_i

        # customer consumption heterogeneity `w_i_cons`
        if "w_i_cons" not in self.data_i:
            self._log("add w_i_cons to data_i")
            self.delta_W_cons = delta_W_cons
            self.w_i_cons_min = w_i_cons_min
            self.w_i_cons_max = w_i_cons_max
            self.data_i["w_i_cons"] = delta_W_cons * np.random.uniform(
                self.w_i_cons_min, self.w_i_cons_max, self.I
            )

        # latent consumer tastes `W_i`
        if W_i is None:
            self._log("build W_i")
            self.W_i = np.random.multivariate_normal(
                np.zeros(self.n_hidden),
                np.identity(self.n_hidden) / self.n_hidden,
                self.I,
            )
        else:
            self.W_i = W_i

        # beta_i_p = price sensitivity
        if data_beta_i_p is None:
            self._log("build data_beta_i_p")
            self.price_sensitivity_levels = price_sensitivity_levels
            if self.price_sensitivity_levels is None:

                mean_ps = mu_ps
                var_ps = sigma_ps
                self.mu_ps = np.log(
                    (mean_ps * mean_ps) / np.sqrt(var_ps + mean_ps * mean_ps)
                )  # mu_p
                # np.exp(self.mu_ps+self.sigma_ps**2/2) == mean_ps
                self.sigma_ps = np.sqrt(
                    np.log(var_ps / (mean_ps * mean_ps) + 1)
                )  # sigma_p
                # (np.exp(self.sigma_ps**2)-1)*np.exp(2*self.mu_ps+self.sigma_ps**2) ==
                # var_ps

                self.data_beta_i_p = pd.DataFrame(
                    {
                        "i": list(range(self.I)),
                        "beta_i_p": -np.random.lognormal(
                            self.mu_ps, self.sigma_ps, self.I
                        ),
                    }
                )
            else:
                self.data_beta_i_p = pd.DataFrame(
                    {
                        "i": list(range(self.I)),
                        "beta_i_p": np.array(self.price_sensitivity_levels)[
                            pd.cut(range(self.I), 4, labels=False)
                        ],
                    }
                )
        else:
            assert self.price_sensitivity_levels is None
            self.data_beta_i_p = data_beta_i_p

        ## C ##

        # category-level data
        if data_c is None:
            self.data_c = pd.DataFrame(
                {
                    "c": list(range(self.C)),
                }
            )
        else:
            self.data_c = data_c

        # add variables to data_c ...
        # ... category consumption rate `cons_c`
        if "cons_c" not in self.data_c:
            self._log("add cons_c to data_c")
            self.cons_c_min = cons_c_min
            self.cons_c_max = cons_c_max
            self.data_c["cons_c"] = np.random.uniform(
                self.cons_c_min, self.cons_c_max, self.C
            )

        # ... base utility for purchasing in category c `gamma_c`
        if "gamma_c" not in self.data_c:
            self._log("add gamma_c to data_c")
            self.gamma_c_min = gamma_c_min
            self.gamma_c_max = gamma_c_max
            self.data_c["gamma_c"] = np.random.uniform(
                self.gamma_c_min, self.gamma_c_max, self.C
            )

        ## ... base price product products in category c `p_c`
        if "p_c" not in self.data_c:
            self._log("add p_c to data_c")
            mean_p = mu_p
            var_p = sigma_p
            self.mu_p = np.log(
                (mean_p * mean_p) / np.sqrt(var_p + mean_p * mean_p)
            )  # mu_p
            self.sigma_p = np.sqrt(np.log(var_p / (mean_p * mean_p) + 1))  # sigma_p
            self.data_c["p_c"] = np.random.lognormal(self.mu_p, self.sigma_p, self.C)

        # ... sensitivity for purchasing in category c given current inventory
        # `gamma_c_inv`
        if "gamma_c_inv" not in self.data_c:
            self._log("add gamma_c_inv to data_c")
            mean_c_inv = gamma_c_inv_mu
            var_c_inv = gamma_c_inv_sigma

            self.gamma_c_inv_mu = np.log(
                (mean_c_inv * mean_c_inv) / np.sqrt(var_c_inv + mean_c_inv * mean_c_inv)
            )  # mu_p
            self.gamma_c_inv_sigma = np.sqrt(
                np.log(var_c_inv / (mean_c_inv * mean_c_inv) + 1)
            )  # sigma_p

            self.data_c["gamma_c_inv"] = -np.random.lognormal(
                self.gamma_c_inv_mu, self.gamma_c_inv_sigma, self.C
            )

        ## J ##

        # product-level data (product dictionary)
        self.data_j = data_j

        assert self.data_j.groupby("c").j.count().nunique() == 1
        self.n_j_by_c = self.J // self.C
        # the following test is really important. we use numpy operations to imrpove speed
        # and this assumes that all categories have the same size!
        assert np.all(self.data_j.groupby("c").j.count().values == self.n_j_by_c)

        if "p_jc" not in self.data_j:
            self.data_j = self.data_j.merge(self.data_c[["c", "p_c"]], on="c")
            self.data_j["price_adj"] = np.random.uniform(low=0.8, high=1.2, size=self.J)
            self.data_j["p_jc"] = self.data_j.p_c * self.data_j.price_adj

        self.data_j = self.data_j[["j", "c", "p_jc"]].copy()

        ## IC ##
        # user-category-level data
        # level for M2

        # define index that is applied to data_ic in the end
        index_ic = ["i", "c"]

        # inventory at t0
        if _inventory_0 is None:
            self.lambda_0_inv = lambda_0_inv
            self._inventory_0 = np.random.exponential(
                self.lambda_0_inv, self.I * self.C
            )
        else:
            self._inventory_0 = _inventory_0

        # m2 model table
        self.data_ic = pd.DataFrame(
            {
                "i": np.repeat(list(range(self.I)), self.C, axis=0),
                "c": list(range(self.C)) * self.I,
                "inv": self._inventory_0,
                "bar_d_ict": 0.0,
                "m2_v_ic": 0.0,
                "m2_u_ic": 0.0,
                "m2_sampled_y": 0,
            }
        )
        self._test_index_ic()

        # user-category consumption `cons_ic`
        if cons_ic is None:
            self._log("build cons_ic")
            self.delta_cons = delta_cons
            self.data_ic = self.data_ic.merge(
                self.data_c[["c", "cons_c"]], on="c", how="left"
            )
            self.data_ic = self.data_ic.merge(
                self.data_i[["i", "w_i_cons"]], on="i", how="left"
            )
            self.data_ic["delta_cons"] = self.delta_cons
            self.data_ic.eval("cons_ic=delta_cons*(1+w_i_cons)*cons_c", inplace=True)
            self.data_ic.drop(
                ["delta_cons", "cons_c", "w_i_cons"], axis=1, inplace=True
            )
        else:
            self.cons_ic = cons_ic
            assert np.all(self.data_ic["i"] == self.cons_ic["i"])
            assert np.all(self.data_ic["c"] == self.cons_ic["c"])
            self.data_ic["cons_ic"] = self.cons_ic["cons_ic"].values
        self._test_index_ic()

        # latent category vectors `Gamma_c`
        if Gamma_c is None:
            self._log("build Gamma_c")
            self.scale_Gamma_c = scale_Gamma_c
            self.Gamma_c = np.random.multivariate_normal(
                np.zeros(self.n_hidden),
                self.scale_Gamma_c * np.identity(self.n_hidden),
                self.C,
            )
        else:
            self.Gamma_c = Gamma_c
        self._test_index_ic()

        # user-category base preferences `gamma_ic`
        if gamma_ic is None:
            self._log("build gamma_ic")
            self.data_ic["gamma_ic"] = self.W_i.dot(self.Gamma_c.T).flatten()
        else:
            self.gamma_ic = gamma_ic
            assert np.all(self.data_ic["i"] == self.gamma_ic["i"])
            assert np.all(self.data_ic["c"] == self.gamma_ic["c"])
            self.data_ic["gamma_ic"] = self.gamma_ic["gamma_ic"].values
            assert np.all(
                np.abs(
                    self.W_i.dot(self.Gamma_c.T).flatten() - self.data_ic["gamma_ic"]
                )
                < 1e-12
            )
        self._test_index_ic()

        # category own-price effect `gamma_ic_p`
        if gamma_ic_p is None:
            self._log("build gamma_ic_p")

            mean_gamma_p = mu_gamma_p
            var_gamma_p = sigma_gamma_p

            self.mu_gamma_p = np.log(
                (mean_gamma_p * mean_gamma_p)
                / np.sqrt(var_gamma_p + mean_gamma_p * mean_gamma_p)
            )  # mu_p
            self.sigma_gamma_p = np.sqrt(
                np.log(var_gamma_p / (mean_gamma_p * mean_gamma_p) + 1)
            )  # sigma_p

            self.gamma_ic_p = np.random.lognormal(
                self.mu_gamma_p, self.sigma_gamma_p, self.I * self.C
            )
            self.data_ic["gamma_ic_p"] = self.gamma_ic_p
        else:
            self.gamma_ic_p = gamma_ic_p
            assert np.all(self.data_ic["i"] == self.gamma_ic_p["i"])
            assert np.all(self.data_ic["c"] == self.gamma_ic_p["c"])
            self.data_ic["gamma_ic_p"] = self.gamma_ic_p["gamma_ic_p"].values
        self._test_index_ic()

        # total cross-price effects (later merged in m2 given current average prices)
        self.data_ic["gamma_ick_cp_total"] = 0.0

        # merge coefficients (for convenience)
        self.data_ic = self.data_ic.merge(
            self.data_c[["c", "gamma_c", "gamma_c_inv"]], on="c"
        )
        self.data_ic = self.data_ic.sort_values(["i", "c"])
        self.data_ic["gamma_c_inv_adj"] = np.random.uniform(
            low=0.9, high=1.1, size=self.I * self.C
        )
        self.data_ic["gamma_ic_inv"] = (
            self.data_ic.gamma_c_inv * self.data_ic.gamma_c_inv_adj
        )
        self._test_index_ic()

        ## IJ ##
        # user-product-level data
        # level for M3 and M4

        # define index that is applied to data_ij in the end
        index_ij = ["i", "j"]

        self.data_ij = pd.DataFrame(
            {
                "i": np.repeat(list(range(self.I)), self.J, axis=0),
                "j": list(range(self.J)) * self.I,
                "m3_sampled_y": 0,
                "m4_sampled_y": 0,
                "d_ijt": 0,
            }
        )
        self._test_index_ij()
        self.unique_is = set(self.data_ij.i.unique())
        self.unique_js = set(self.data_ij.j.unique())

        self._precomputed_gumbel = np.random.gumbel(0, 1, self.data_ij.shape[0])

        # latent product vectors `Beta_jc`
        if Beta_jc is None:
            self._log("build Beta_c")
            # mean of latent product vectors `mu_Beta_jc`
            if mu_Beta_jc is None:
                self.mu_Beta_jc_min = mu_Beta_jc_min
                self.mu_Beta_jc_max = mu_Beta_jc_max
                self.mu_Beta_jc = np.random.uniform(
                    self.mu_Beta_jc_min, self.mu_Beta_jc_max, self.n_hidden
                )
            else:
                self.mu_Beta_jc = mu_Beta_jc

            self.scale_Beta_jc = scale_Beta_jc
            self.Beta_jc = self.scale_Beta_jc * np.random.multivariate_normal(
                np.zeros(self.n_hidden),
                np.identity(
                    self.n_hidden
                ),  # self.scale_Beta_jc * np.identity(self.n_hidden),
                self.J,
            )
        else:
            self.Beta_jc = Beta_jc

        Beta_jc_offset = np.random.uniform(
            self.mu_Beta_jc_min, self.mu_Beta_jc_max, self.J
        )

        # user-product base preferences `beta_ijc`
        if beta_ijc is None:
            self._log("build beta_ijc")
            beta_ijc_values = self.W_i.dot(self.Beta_jc.T).flatten()
            # beta_ijc_values = self.W_i.dot(self.Beta_jc.T).T.flatten()
            # np.random.shuffle(beta_ijc_values)
            self.data_ij["beta_ijc"] = beta_ijc_values
        else:
            self.beta_ijc = beta_ijc
            assert np.all(self.data_ij["i"] == self.beta_ijc["i"])
            assert np.all(self.data_ij["j"] == self.beta_ijc["j"])
            self.data_ij["beta_ijc"] = self.beta_ijc["beta_ijc"].values
        self._test_index_ij()

        # merge price sensitivity
        self.data_ij = self.data_ij.merge(self.data_beta_i_p, on="i", how="left")
        self._test_index_ij()

        # product base utility
        if "beta_j" not in self.data_j:
            self.data_j["beta_j"] = Beta_jc_offset
        self._test_index_ij()

        # merge product data (for convenience)
        self.data_ij = pd.merge(
            self.data_ij, self.data_j[["c", "j", "p_jc", "beta_j"]], on=["j"]
        )
        self.data_ij = self.data_ij.sort_values(["i", "j"])
        self._test_index_ij()

        ## FORMATTING
        self.data_i.set_index(["i"], inplace=True)
        self.data_ic.set_index(index_ic, inplace=True)
        self.data_ic.sort_index(inplace=True)
        self.data_ic["idx_ic"] = list(range(self.data_ic.shape[0]))

        self.data_ij = self.data_ij.merge(self.data_ic[["idx_ic"]], on=["i", "c"])
        self.data_ij.set_index(index_ij, inplace=True)
        self.data_ij.sort_index(inplace=True)
        self.data_ij["idx_ij"] = list(range(self.data_ij.shape[0]))

        self.data_ij_idx = self.data_ij.reset_index()[["i", "j"]]

        ## ICK ##
        # category's cross-price effect note that coefficients for own-price effects are 0
        # and that effects are symmetric

        self.data_ic_index = self.data_ic.reset_index()[["i", "c"]]

        self.cp_type = cp_type
        self.delta_cp = delta_cp
        self.sigma_cp = sigma_cp
        self.prob_cp = prob_cp
        self.min_mu_cp = min_mu_cp
        self.max_mu_cp = max_mu_cp
        self.gamma_ick_cp = np.zeros((self.I * self.C, self.C))
        for c in range(self.C):
            for k in range(self.C):
                if c > k:
                    if self.cp_type == "log-normal":
                        values_ck = np.random.choice(
                            [0, 1], p=[1 - self.prob_cp, self.prob_cp]
                        ) * np.random.lognormal(
                            self.delta_cp
                            * np.random.uniform(self.min_mu_cp, self.max_mu_cp, 1),
                            self.sigma_cp,
                            self.I,
                        )
                    elif self.cp_type == "normal":
                        values_ck = (
                            np.random.choice([0, 1], p=[1 - self.prob_cp, self.prob_cp])
                            * np.random.normal(
                                self.delta_cp * np.random.uniform(-1, 1, 1), 1, self.I
                            )
                            * self.sigma_cp
                        )
                    elif self.cp_type == "sign-log-normal":
                        mean_lognormal = self.delta_cp
                        var_lognormal = self.sigma_cp
                        mu_lognormal = np.log(
                            (mean_lognormal * mean_lognormal)
                            / np.sqrt(var_lognormal + mean_lognormal * mean_lognormal)
                        )
                        sigma_lognormal = np.sqrt(
                            np.log(
                                var_lognormal / (mean_lognormal * mean_lognormal) + 1
                            )
                        )

                        p_sign = [1 - self.prob_cp, self.prob_cp / 2, self.prob_cp / 2]
                        values_ck = np.random.choice(
                            [0, 1, -1], p=p_sign
                        ) * np.random.lognormal(mu_lognormal, sigma_lognormal)
                    else:
                        raise Exception(
                            "cp_type not supported, use `log-normal` or `normal`"
                        )
                    self.gamma_ick_cp[self.data_ic_index["c"] == c, k] = values_ck
                    self.gamma_ick_cp[self.data_ic_index["c"] == k, c] = values_ck

        # compute base probabilities for product choice
        # make sure that there is no discount
        self.data_ij["price_paid"] = self.data_ij["p_jc"]
        # compute and save probabilities
        self._m3_sample()
        self.data_ij["m3_p_ijt_base"] = self.data_ij["m3_p_ijt"]

        # tests
        assert self.data_ic.index.identical(
            self.data_ij.reset_index()[["i", "c"]]
            .drop_duplicates()
            .set_index(["i", "c"])
            .index
        )

        ## logging
        self._log(self.data_i)
        self._log(self.data_c)
        self._log(self.data_j)
        self._log(self.data_ic)
        self._log(self.data_j.groupby("c")["j"].count().min())
        self._log("setup done")

        ## BURN-IN

        if self.do_burn_in:
            self._run_burn_in(burnin_action)

    ## main function for generating basket data
    def generate(self, T=1, action=None, seed_offset=0):

        """Generate basket data

        Args:
            T (int): number of weeks to generate baskets for
            action (DataFrame): coupon intervention, a pandas DataFrame with coupon treatment--must contain
                columns i (individual), j (product), t (week/time), and discount (price-off in [0, 1])
            seed_offset (int): offset for RNG
        """

        # test action
        if isinstance(action, pd.DataFrame):
            assert len(set(action.i.unique()).difference(self.unique_is)) == 0
            assert len(set(action.j.unique()).difference(self.unique_js)) == 0
            assert (
                action.groupby(["i", "j", "t"] if "t" in action else ["i", "j"])
                .discount.count()
                .max()
                == 1
            )

        # generate baskets
        basket_list = []

        for t in self.week + np.arange(T):

            # set week-specific seed
            np.random.seed(501 + self.week + seed_offset)

            # update inventory: purchases + consumption
            # the probabilities would be the same if we update the inventory at the end of the loop but it's
            # better to do it here because then self.data_ij contains the inv that was used to compute
            # m2_sampled_y -- otherwise inv and m2_sampled_y are not "synced"
            # first purchases (+), then consumption (-) because 0 is lower bound for inventory
            self._inventory_apply_purchases()
            self._inventory_apply_consumption()

            # add discounts to data_ij
            self._merge_action(action, t=t)

            # update price paid
            self.data_ij["price_paid"] = self.data_ij["p_jc"] * (
                1 - self.data_ij["d_ijt"]
            )

            # m3: product choice model
            # sample products users purchase in t (condition on purchase in category)
            # discrete choice, i.e. exactly one product per (i, c)
            self._m3_sample()

            # m2: category purchase incidence model
            # sample categories users purchase in t (conditioned on shopping trip)
            self._m2_sample()

            # m4: purchase quantity model
            self._m4_sample()

            # compile basket data
            if self.data_ij["m3_sampled_y"].sum() > 0:
                basket_data_t = self.data_ij[
                    self.data_ij["m3_sampled_y"] > 0
                ].reset_index()
                basket_data_t = basket_data_t[["i", "j", "p_jc", "price_paid", "d_ijt"]]
                basket_data_t["t"] = t
                basket_list.append(basket_data_t)

            # update week
            self.week += 1

        return pd.concat(basket_list).reset_index(drop=True)

    def recreate_gym_state(self, x, baskets, target_week):
        """Recreate gym state between week 0 and

        Args:
            x (SupermarketGym): gym object in week 0
            baskets (DataFrame): baskets starting in week 0, at least to week `target_week`
            target_week (int): the week of output gym state
        """

        assert x.week == 0

        # only works for (binary) purchase events
        purchases_ict = baskets.merge(x.data_j, on="j")[
            ["i", "c", "t"]
        ].drop_duplicates()
        purchases_ict["m2_sampled_y"] = 1

        for t in range(target_week):
            # 1. update inventory
            x._inventory_apply_purchases()
            x._inventory_apply_consumption()

            # 2. purchases
            purchases_ic_t = purchases_ict[purchases_ict["t"] == t]
            x.data_ic["m2_sampled_y"] = (
                self.data_ic_index.merge(purchases_ic_t, on=["i", "c"], how="left")
                .m2_sampled_y.fillna(0)
                .astype(int)
                .values
            )

            # 3. update week
            x.week += 1

        return x

    def extract_data_from_gym(self, x, baskets, set_is, target_week, W, action=None):
        """Extract probability and debug data from gym

        At a given point in time, for a specified number of weeks, and for a specified user set

        Args:
            x (SupermarketGym): gym object in week 0
            baskets (DataFrame): baskets starting in week 0, at least to week `target_week`
            set_is (list): set of users for which data should be extracted
            target_week (int): the week of output gym state
            W (int): number of weeks (starting with target week) for which data should be extracted
        """

        x2 = self.recreate_gym_state(x=x, baskets=baskets, target_week=target_week)

        _tmp_baskets = []
        _tmp_data_ic = []
        _tmp_data_ij = []

        for t in range(W):
            baskets_t = x2.generate(T=1, action=action)
            _tmp_baskets.append(baskets_t[baskets_t["i"].isin(set_is)])

            data_ic_t = x2.data_ic[
                ["m2_u_ic", "m2_v_ic", "inv", "eps_ict", "gamma_ick_cp_total"]
            ].reset_index()
            data_ic_t["t"] = baskets_t["t"].values[0]
            data_ic_t["m2_p_ict"] = scipy.stats.norm.cdf(data_ic_t["m2_v_ic"])
            _tmp_data_ic.append(data_ic_t[data_ic_t["i"].isin(set_is)])

            data_ij_t = x2.data_ij[
                [
                    "c",
                    "m3_p_ijt",
                    "m3_u_ijt",
                    "m3_v_ijt",
                    "eps_ijt",
                    "d_ijt",
                    "price_paid",
                ]
            ].reset_index()
            data_ij_t["t"] = baskets_t["t"].values[0]
            _tmp_data_ij.append(data_ij_t[data_ij_t["i"].isin(set_is)])

        return pd.concat(_tmp_baskets), pd.concat(_tmp_data_ic), pd.concat(_tmp_data_ij)

    def save(self, f, keep_data_burnin=False):
        if not keep_data_burnin:
            self.data_burnin = None
        con = open(f, "wb")
        pickle.dump(self, con, protocol=4)
        con.close()

    @staticmethod
    def load(f):
        f = open(f, "rb")
        gym = pickle.load(f)
        f.close()
        return gym

    # model functions >>>
    def _m2_sample(self):
        # sample error term
        self.data_ic["eps_ict"] = np.random.normal(0, 1, size=(self.I * self.C))

        # compute average price for each user-category combination ...
        # ... and merge it to self.data_ic
        if self.own_price_method == "probability":
            _tmp_bar_d_ict = np.sum(
                (
                    self.data_ij["m3_p_ijt_base"].values * self.data_ij["d_ijt"].values
                ).reshape(-1, self.n_j_by_c, order="C"),
                axis=1,
            )
        elif self.own_price_method == "choice":
            _tmp_bar_d_ict = np.sum(
                (
                    self.data_ij["m3_sampled_y"].values * self.data_ij["d_ijt"].values
                ).reshape(-1, self.n_j_by_c, order="C"),
                axis=1,
            )
        elif self.own_price_method == "mean":
            _tmp_bar_d_ict = np.mean(
                self.data_ij["d_ijt"].values.reshape(-1, self.n_j_by_c, order="C"),
                axis=1,
            )
        else:
            raise Exception("own_price_method not supported")
        self.data_ic["bar_d_ict"] = _tmp_bar_d_ict

        # compute utility from cross-price effects
        self.data_ic["gamma_ick_cp_total"] = np.sum(
            self.gamma_ick_cp
            * self.data_ic["bar_d_ict"].values.reshape(-1, self.C)[
                np.repeat(range(self.I), self.C)
            ],
            axis=1,
        )

        # total utilities, probabilities, and choices
        self.data_ic["m2_v_ic"] = (
            self.data_ic["gamma_ic_inv"] * self.data_ic["inv"]
            + self.data_ic["gamma_ic_p"] * self.data_ic["bar_d_ict"]
            + self.data_ic["gamma_ick_cp_total"]
            + self.data_ic["gamma_c"]
            + self.data_ic["gamma_ic"]
        )
        self.data_ic["m2_u_ic"] = self.data_ic["m2_v_ic"] + self.data_ic["eps_ict"]
        self.data_ic["m2_sampled_y"] = (self.data_ic["m2_u_ic"] > 0).astype(int)

        # Corrected Line: Use .iloc for integer-based indexing
        self.data_ij["m3_sampled_y"] = (
            self.data_ij.m3_sampled_y.values
            * self.data_ic.m2_sampled_y.iloc[self.data_ij.idx_ic].values
        )



    def _m3_sample(self):
        # straight-forward logit
        _copy_precomputed_gumbel = self._precomputed_gumbel.copy()
        np.random.shuffle(_copy_precomputed_gumbel)
        self.data_ij["eps_ijt"] = _copy_precomputed_gumbel
        self.data_ij["m3_v_ijt"] = (
            self.data_ij["beta_j"]
            + self.data_ij["beta_ijc"]
            + self.data_ij["beta_i_p"] * self.data_ij["price_paid"]
        )
        self.data_ij["m3_u_ijt"] = self.data_ij["m3_v_ijt"] + self.data_ij["eps_ijt"]
        # for each (i,c): choice = argmax(u_ijt)
        arr_u_ijt = self.data_ij.m3_u_ijt.values.reshape(-1, self.n_j_by_c)
        arr_sampled_y = np.zeros_like(arr_u_ijt, dtype=int)
        arr_sampled_y[np.arange(arr_u_ijt.shape[0]), np.argmax(arr_u_ijt, axis=1)] = 1
        self.data_ij["m3_sampled_y"] = arr_sampled_y.flatten()
        _m3_exp_v_ijt = np.exp(self.data_ij["m3_v_ijt"])
        _m3_sum_exp_v_ijt = np.repeat(
            np.sum(_m3_exp_v_ijt.values.reshape(-1, self.n_j_by_c), axis=1),
            self.n_j_by_c,
        )
        self.data_ij["m3_p_ijt"] = _m3_exp_v_ijt / _m3_sum_exp_v_ijt

    def _m4_sample(self):
        # currently no quantity choice
        self.data_ij["m4_sampled_y"] = self.data_ij["m3_sampled_y"]

    # helper functions >>>

    # self.generate wrapper for simulation burn in
    def _run_burn_in(self, burnin_action):
        self._log("start burnin")
        if self.burn_in > 0:
            self._log("initialize data (burn in)")
            _tmp_data_burnin = []

            for t_burnin in range(-(self.burn_in), 0):
                _tmp_data_burnin.append(self.generate(T=1, action=burnin_action))

            if self.save_data_burnin:
                self.data_burnin = pd.concat(_tmp_data_burnin)
            else:
                self.data_burnin = None

            self._log("burn in done")

        else:
            self._log("WARN -- no burn in (self.burn_in = 0)")

    def _inventory_apply_consumption(self):
        self.data_ic["inv"] = (self.data_ic["inv"] - self.data_ic["cons_ic"]).clip(
            lower=0
        )

    def _inventory_apply_purchases(self):
        self.data_ic["inv"] += self.data_ic["m2_sampled_y"]

    def _merge_action(self, action, t):
        self.data_ij["d_ijt"] = 0.0
        if isinstance(action, pd.DataFrame):
            # IF t is defined     subset actions to t=current week (self.week+np.arange(T))
            # ELSE                use all coupons in action for current week -- note that this will
            #                     generate a warning if T>1 because coupons are recycled every week
            if "t" in action:
                action_t = action[action["t"] == t]
            else:
                action_t = action
            if action_t.shape[0] > 0:
                _tmp_discounts = np.zeros(self.data_ij.shape[0])
                _tmp_discounts[action_t.i * self.J + action_t.j] = action_t.discount
                self.data_ij["d_ijt"] = _tmp_discounts

    def _test_index_ij(self):
        assert np.all(
            self.data_ij.reset_index()["i"]
            == np.repeat(list(range(self.I)), self.J, axis=0)
        )
        assert np.all(self.data_ij.reset_index()["j"] == list(range(self.J)) * self.I)

    def _test_index_ic(self):
        assert np.all(
            self.data_ic.reset_index()["i"]
            == np.repeat(list(range(self.I)), self.C, axis=0)
        )
        assert np.all(self.data_ic.reset_index()["c"] == list(range(self.C)) * self.I)

    def _test_index_ij(self):
        assert np.all(
            self.data_ij.reset_index()["i"]
            == np.repeat(list(range(self.I)), self.J, axis=0)
        )
        assert np.all(self.data_ij.reset_index()["j"] == list(range(self.J)) * self.I)

    def _log(self, x):
        # only log if verbose > 1
        # IF        verbose == 1   only log text
        # ELSE IF   verbose > 1    also log other data types (e.g., np.array, pd.DataFrame, etc.)
        if self.verbose > 0:
            if self.verbose > 1 or isinstance(x, str):
                logger.info(x)
