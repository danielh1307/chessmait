import torch
import torch.nn as nn


class CustomWeightedMSELoss(nn.Module):
    def __init__(self):
        super(CustomWeightedMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-6  # To avoid division by zero
        # consider this alternative
        # weights = torch.clamp(weights, max=10)

        # this worked well, but the penalty is for y_true next to 0:
        # weights = torch.log1p(1 / (torch.abs(y_true) + epsilon))

        # give more penalty when y_true is next to 0.5:
        weights = 1 - torch.abs(y_true - 0.5) * 2
        weights = torch.log1p(1 / (weights + epsilon))

        loss = weights * (y_pred - y_true) ** 2
        return torch.mean(loss)
