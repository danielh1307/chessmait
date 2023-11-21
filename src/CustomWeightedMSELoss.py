import torch
import torch.nn as nn


# This loss is intended to "punish" those losses more which are close to 0.5,
# this is close to "the middle" when normalizing values between 0 and 1.
# A difference of 0.5 pawns is worse if it is between 0 and 0.5 than it is between
# 3.5 and 4.
class CustomWeightedMSELoss(nn.Module):
    def __init__(self):
        super(CustomWeightedMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-6  # To avoid division by zero

        # give more penalty when y_true is next to 0.5:
        weights = 1 - torch.abs(y_true - 0.5) * 2

        # just using the weights did not let the models converge;
        # that's why we are using log1p here
        weights = torch.log1p(1 / (weights + epsilon))
        loss = weights * (y_pred - y_true) ** 2

        return torch.mean(loss)
