# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import torch
import torch.nn as nn
import numpy as np

from loguru import logger


class ModelBase(nn.Module):
    def __init__(self, pretrained=None):
        super(ModelBase, self).__init__()
        self.pretrained = pretrained.copy() if pretrained is not None else None

    def save_weights(self, path, epoch):
        torch.save(self.state_dict(), "%s/results/state_dict_%08d.pt" % (path, epoch))

    def load_weights(self):

        # load pretrained weights
        logger.info(f"Load weights from file {self.pretrained['file']}")
        pretrained_state_dict = torch.load(self.pretrained["file"])

        # if weights are not specified: find out which weights should be loaded
        if self.pretrained["weights"] is None:
            weights_pretrained = set(pretrained_state_dict.keys())
            weights_model = set(self.state_dict().keys())
            self.pretrained["weights"] = list(
                weights_pretrained.intersection(weights_model)
            )
        # else: check whether all specified weights are available
        else:
            assert np.all(
                [x in pretrained_state_dict for x in self.pretrained["weights"]]
            )

        # build weights map
        weights_map = {}
        for v in self.pretrained["weights"]:
            weights_map[v] = v
        # add custom weight map
        custom_weight_map = self.pretrained["custom_weight_map"]
        if custom_weight_map is not None:
            assert np.all(
                [x in pretrained_state_dict for x in set(custom_weight_map.values())]
            )
            assert np.all(
                [x in self.state_dict().keys() for x in set(custom_weight_map.keys())]
            )
            weights_map = {**weights_map, **custom_weight_map}
        for k, v in weights_map.items():
            logger.info(f"initalize `{k}` with `{v}`")

        # load weights into model
        weights_to_load = {k: pretrained_state_dict[v] for k, v in weights_map.items()}
        dnn_model_state_dict = self.state_dict()
        dnn_model_state_dict.update(weights_to_load)
        self.load_state_dict(dnn_model_state_dict)
