# source from https://github.com/rssalessio/PytorchRBFLayer/blob/main/rbf_layer/rbf_layer.py

import torch
import torch.nn as nn
import src.model.rbf1 as rbf1


class RBFLayer2(nn.Module): #Radial Basis Function Layer

    def __init__(self, in_features, out_features, kernel, radial_function, norm_function):
        super(RBFLayer2, self).__init__()

        self.in_features_dim = in_features
        self.num_kernels = kernel
        self.out_features_dim = out_features
        self.radial_function = radial_function
        self.norm_function = norm_function
        self.normalization = True

        self._make_parameters()

    def _make_parameters(self) -> None:
        # Initialize linear combination weights
        self.weights = nn.Parameter(
            torch.zeros(
                self.out_features_dim,
                self.num_kernels,
                dtype=torch.float32))

        # Initialize kernels' centers
        self.kernels_centers = nn.Parameter(
            torch.zeros(
                self.num_kernels,
                self.in_features_dim,
                dtype=torch.float32))

        # Initialize shape parameter
        self.log_shapes = nn.Parameter(
            torch.zeros(self.num_kernels, dtype=torch.float32))

        self.reset()

    def reset(self,
              upper_bound_kernels: float = 1.0,
              std_shapes: float = 0.1,
              gain_weights: float = 1.0) -> None:

        nn.init.uniform_(
            self.kernels_centers,
            a=-upper_bound_kernels,
            b=upper_bound_kernels)

        nn.init.normal_(self.log_shapes, mean=0.0, std=std_shapes)

        nn.init.xavier_uniform_(self.weights, gain=gain_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # Input has size B x Fin
        batch_size = input.size(0)

        # Compute difference from centers
        # c has size B x num_kernels x Fin
        c = self.kernels_centers.expand(batch_size, self.num_kernels,
                                        self.in_features_dim)

        diff = input.view(batch_size, 1, self.in_features_dim) - c

        # Apply norm function; c has size B x num_kernels
        r = self.norm_function(diff)

        # Apply parameter, eps_r has size B x num_kernels
        eps_r = self.log_shapes.exp().expand(batch_size, self.num_kernels) * r

        # Apply radial basis function; rbf has size B x num_kernels
        rbfs = self.radial_function(eps_r)

        # Apply normalization
        # (check https://en.wikipedia.org/wiki/Radial_basis_function_network)
        if self.normalization:
            # 1e-9 prevents division by 0
            rbfs = rbfs / (1e-9 + rbfs.sum(dim=-1)).unsqueeze(-1)

        # Take linear combination
        out = self.weights.expand(batch_size, self.out_features_dim,
                                  self.num_kernels) * rbfs.view(
            batch_size, 1, self.num_kernels)
        return out.sum(dim=-1)


def l_norm(x, p=2):
    return torch.norm(x, p=p, dim=-1)


class RBFNet2(nn.Module):
    def __init__(self):
        super(RBFNet2, self).__init__()
        self.rbf = RBFLayer2(768, 128, 3, rbf1.gaussian, l_norm)
        self.linear = nn.Linear(128, 1)

    def forward(self, x):
        x = self.rbf(x)
        x = self.linear(x)
        x = torch.tanh(x)
        return x
