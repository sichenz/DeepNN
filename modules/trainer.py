# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import torch
import torch.nn.functional as F
import torch.optim as optim

import os
import collections
import numpy as np
import pandas as pd

from loguru import logger

import modules.lib as lib
import modules.data_loader as data_loader


class Trainer:
    def __init__(
        self,
        model,
        experiment,
        dataset_train,
        dataset_validation=None,
        save_yaml=True,
        makedir=True,
        use_gpu=False,
    ):

        # dnn
        self.Model = model

        # data
        self.dataset_train = dataset_train
        self.dataset_validation = dataset_validation

        # optimizer
        self.optimizer = optim.Adam(self.Model.parameters())

        # cuda
        self.use_cuda = use_gpu
        if self.use_cuda:
            self.Model.cuda()

        # monitoring
        self.experiment = experiment.copy()
        self.path = self.experiment["trainer"]["path"]
        if makedir:
            os.makedirs(self.path, exist_ok=True)
            os.makedirs(f"{self.path}/results", exist_ok=True)
        if save_yaml:
            lib.write_yaml(self.experiment, f"{self.path}/experiment.yaml")

        self.counter_epoch = 0

        self.n_batch_print_loss = self.experiment["trainer"]["n_batch_print_loss"]
        self.n_epoch_save_results = self.experiment["trainer"]["n_epoch_save_results"]
        self.n_epoch_save_model = self.experiment["trainer"]["n_epoch_save_model"]
        self.l1_lambda = self.experiment["trainer"]["l1_lambda"]
        self.l1_variables = self.experiment["trainer"]["l1_variables"]
        self.l2_lambda = self.experiment["trainer"]["l2_lambda"]
        self.l2_variables = self.experiment["trainer"]["l2_variables"]

    def train(
        self,
        n_epoch=50,
        batch_size=1024,
        batch_size_validation=4096,
        freeze_weights=None,
    ):

        # data loader
        train_loader = data_loader.DataLoaderDNN(
            dataset=self.dataset_train,
            num_workers=0,
            batch_size=1,
            batch_sampler=data_loader.BatchSamplerDNN(
                source=self.dataset_train,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            ),
            pin_memory=True,
        )

        validation_loader = data_loader.DataLoaderDNN(
            dataset=self.dataset_validation,
            num_workers=0,
            batch_size=1,
            batch_sampler=data_loader.BatchSamplerDNN(
                source=self.dataset_validation,
                batch_size=batch_size_validation,
                shuffle=True,
                drop_last=True,
            ),
            pin_memory=True,
        )

        # log training parameters
        self.experiment["train-log"][str(self.counter_epoch)] = {
            "training_parameter": {
                "n_epoch": n_epoch,
                "batch_size": batch_size,
                "batch_size_validation": batch_size_validation,
                "freeze_weights": freeze_weights,
            }
        }

        # freeze weights
        if len(freeze_weights) > 0:
            logger.info(f"Freeze weights [{', '.join(freeze_weights)}]")
            for name, param in self.Model.named_parameters():
                if name in freeze_weights:
                    param.requires_grad = False
        else:
            logger.info("Training all weights")

        # dimensions
        J = self.Model.J
        T = self.experiment["global_streamer_parameters"]["history_length"]

        # train for multiple epochs
        for i in range(n_epoch):

            # EPOCH START >>
            loss_train, loss_validation = collections.deque(), collections.deque()
            self.counter_batch = 0

            # EPOCH MAIN >>
            for labels, discounts, buycounts, frequencies in train_loader:
                # labels, discounts, buycounts, frequencies = train_loader.__iter__().next()
                if labels.shape[0] < batch_size:
                    continue
                self.counter_batch += 1

                buycounts = buycounts.reshape(batch_size, T, J)
                if self.use_cuda:
                    frequencies = frequencies.cuda()
                    discounts = discounts.cuda()
                    buycounts = buycounts.cuda()
                    labels = labels.cuda()

                self.optimizer.zero_grad()
                logits = self.Model(
                    frequencies,
                    discounts,
                    buycounts,
                )
                loss = F.multilabel_soft_margin_loss(logits, labels)
                loss_scalar = loss.item()

                # regularisation
                if self.l1_lambda is not None:
                    loss += self._get_lx_loss(
                        lx_lambda=self.l1_lambda, lx_variables=self.l1_variables, norm=1
                    )
                if self.l2_lambda is not None:
                    loss += self._get_lx_loss(
                        lx_lambda=self.l2_lambda, lx_variables=self.l2_variables, norm=2
                    )

                loss.backward()
                self.optimizer.step()

                # save loss
                loss_train.append(loss_scalar)

                if self.counter_batch % self.n_batch_print_loss == 0:
                    print(np.round(loss_scalar, 4), end=", ")

                # self.Model.eye_j.detach() # detach it
                # self.Model.eye_block_c.detach() # detach it
                torch.cuda.empty_cache()

            # EPOCH END >>
            # save training losses
            np.save(
                os.path.join(
                    self.path, "results", "loss_train_%08d" % self.counter_epoch
                ),
                list(loss_train),
            )
            loss_train = collections.deque()

            # validation loss
            for labels, discounts, buycounts, frequencies in validation_loader:
                buycounts = buycounts.reshape(batch_size_validation, T, J)
                if self.use_cuda:
                    frequencies = frequencies.cuda()
                    discounts = discounts.cuda()
                    buycounts = buycounts.cuda()
                    labels = labels.cuda()
                logits = self.Model(
                    frequencies,
                    discounts,
                    buycounts,
                )
                loss_validation.append(
                    F.multilabel_soft_margin_loss(logits, labels).item()
                )
            np.save(
                os.path.join(
                    self.path, "results", "loss_validation_%08d" % self.counter_epoch
                ),
                loss_validation,
            )
            print("**%.4f**" % np.mean(loss_validation), end=", ")
            loss_validation = collections.deque()

            # save weights
            if (self.counter_epoch % self.n_epoch_save_results) == 0:
                self.Model.save_weights(self.path, self.counter_epoch)
                torch.save(
                    self.optimizer.state_dict(),
                    f"{self.path}/results/optim_state_dict_{self.counter_epoch:08d}.pt",
                )

            self.counter_epoch += 1

        # save training parameters (log training)
        print("\n")
        lib.write_yaml(self.experiment, f"{self.path}/experiment.yaml")

    def predict(self, x, batch_size=2048):
        if batch_size > x.num_training_samples:
            batch_size = x.num_training_samples
        x.reset_streamer(randomize=False)
        probabilities = []
        while batch_size > 0:
            labels, discounts, buycounts, frequencies, _ = x.get_batch(batch_size)
            labels = torch.from_numpy(labels).float()
            discounts = torch.from_numpy(discounts).float()
            buycounts = torch.from_numpy(buycounts).float()
            frequencies = torch.from_numpy(frequencies).float()
            if self.use_cuda:
                labels = labels.cuda()
                discounts = discounts.cuda()
                buycounts = buycounts.cuda()
                frequencies = frequencies.cuda()
            logits = self.Model(frequencies, discounts, buycounts)
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
            batch_size = np.min([batch_size, len(x.user_time_pairs_cache)])
        probabilities_df = self._list_2_df(
            x=probabilities, streamer=x, value_name="phat"
        )
        return probabilities_df

    def _list_2_df(self, x, streamer, value_name):
        x = pd.DataFrame(np.vstack(x))
        x_df = pd.DataFrame(streamer.shuffled_user_time_pairs)
        x_df.columns = ["i", "t"]
        x_df = x_df.join(x)
        x_df = x_df.melt(["i", "t"], var_name="j", value_name=value_name)
        x_df["j"] = x_df["j"].astype(int).map(streamer.id2prod)
        x_df["i"] = x_df["i"].astype(int).map(streamer.id2cust)
        x_df = x_df.sort_values(["i", "t", "j"])
        return x_df

    def _get_lx_loss(self, lx_lambda, lx_variables, norm):
        lx_loss = 0
        for v in lx_variables:
            lx_loss += torch.norm(self.Model.state_dict()[v], norm)
        return lx_lambda * lx_loss
