# source https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer

import torch
import torch.nn as nn


class RBFLayer1(nn.Module):

    def __init__(self, in_features, out_features, basic_func):
        super(RBFLayer1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
        self.basis_func = basic_func
        self.reset_parameters()
        self.float()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        r = self.basis_func(distances)
        return r


# RBFs
def gaussian(alpha):
    phi = torch.exp(-1 * alpha.pow(2))
    return phi


def linear(alpha):
    phi = alpha
    return phi


def quadratic(alpha):
    phi = alpha.pow(2)
    return phi


def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


class RBFNet1(nn.Module):
    def __init__(self):
        super(RBFNet1, self).__init__()
        self.layer1 = nn.Sequential(
            RBFLayer1(768, 128, gaussian),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.output_layer(x)
        return x