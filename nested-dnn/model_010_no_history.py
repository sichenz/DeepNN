# Gabel and Timoshenko (2021), "Product Choice with Large Assortments: A Scalable Deep-Learning Model." Management Science, Forthcoming

import torch
import torch.nn as nn

from modules.model_base import ModelBase


class Model(ModelBase):
    """
    model 010 = model 005 + different embeddings for bottleneck input and output
    """

    def __init__(self, J, T, K, L, epsilon, pretrained):

        # init
        ModelBase.__init__(self, pretrained)
        self.epsilon = epsilon
        self.J = J

        # time embedding
        self.w_conv_t = nn.Parameter(torch.FloatTensor(T, K).uniform_(0.18, 0.22))

        # product embedding
        self.w_conv_j = nn.Parameter(torch.FloatTensor(J, L).uniform_(-0.025, 0.025))
        self.w_conv_j2 = nn.Parameter(torch.FloatTensor(J, L).uniform_(-0.025, 0.025))
        self.w_conv_j_d = nn.Parameter(torch.FloatTensor(J, L).uniform_(-0.025, 0.025))
        self.w_conv_j_d2 = nn.Parameter(torch.FloatTensor(J, L).uniform_(-0.025, 0.025))
        self.w_conv_j_pf = nn.Parameter(torch.FloatTensor(J, L).uniform_(-0.025, 0.025))
        self.w_conv_j_pf2 = nn.Parameter(
            torch.FloatTensor(J, L).uniform_(-0.025, 0.025)
        )

        # frequency embedding
        self.w_pf_filter = nn.Parameter(torch.FloatTensor(K).uniform_(0.5, 0.7))

        # output weights
        self.w_out_conv_t = nn.Parameter(torch.FloatTensor(K, 1).uniform_(-0.25, 0.25))
        self.w_out_conv_j = nn.Parameter(torch.FloatTensor(K, 1).uniform_(-0.1, 0.1))
        self.w_out_discount = nn.Parameter(torch.FloatTensor(J).uniform_(0.1, 0.2))
        self.w_out_discount_cross = nn.Parameter(
            torch.FloatTensor(J).uniform_(0.1, 0.2)
        )

        # bias
        self.ff_out_b = nn.Parameter(torch.FloatTensor(1, J, 1).uniform_(-3, -2.5))

        # load pretrained weights
        if pretrained is not None:
            self.load_weights()

    def forward(self, in_pf, in_np, in_bc):

        # dimensions
        B = in_pf.shape[0]
        J = in_pf.shape[1]

        # purchase frequency
        clamp_pf = torch.clamp(in_pf, self.epsilon, 1 - self.epsilon)
        logit_pf = torch.log(clamp_pf / (1 - clamp_pf))

        # purchase frequency cross
        logit_pf_cross_in = torch.einsum("bj,jl->bl", logit_pf, self.w_conv_j_pf)
        logit_pf_cross = torch.einsum("bl,jl->bj", logit_pf_cross_in, self.w_conv_j_pf2)

        # discount
        logit_discount = torch.mul(in_np, self.w_out_discount)

        # discount cross
        cross_price_in = torch.einsum("bj,jl->bl", in_np, self.w_conv_j_d)
        cross_price_out = torch.einsum("bl,jl->bj", cross_price_in, self.w_conv_j_d2)
        logit_discount_cross = torch.mul(cross_price_out, self.w_out_discount_cross)

        return (
            torch.squeeze(self.ff_out_b, 2)
            + logit_pf
            + logit_pf_cross
            + logit_discount
            + logit_discount_cross
        )
