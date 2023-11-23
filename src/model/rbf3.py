# source from https://github.com/amanuelanteneh/CFF-Net/tree/main

import torch
import torch.nn as nn
import numpy as np


class RBFLayer3(nn.Module):
    def __init__(self, inFeatures, outFeatures, centers, useAvgDist, device):
        super(RBFLayer3, self).__init__()
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.centers = torch.from_numpy(centers).float()
        self.centers = self.centers.to(device)
        # calculate distance between any two cluster centers in centers variable
        clusterDistances = [np.linalg.norm(c1 - c2) for c1 in centers for c2 in centers]
        dMax = max(clusterDistances)
        dAvg = sum(clusterDistances) / len(clusterDistances)
        # define tensor of sigmas of dimension 1 x len(centers)
        if useAvgDist == True:
            self.sigma = torch.full((1, len(self.centers)), (2 * dAvg), device=device)
        else:
            self.sigma = torch.full((1, len(self.centers)), (np.sqrt(dMax) / len(self.centers)), device=device)

    def forward(self, input):
        # Input has shape batchSize x inFeature
        batchSize = input.size(0)

        mu = self.centers.view(len(self.centers), -1).repeat(batchSize, 1, 1)
        X = input.view(batchSize, -1).unsqueeze(1).repeat(1, len(self.centers), 1)

        # Gaussian RBF
        # Phi = torch.exp( -(torch.pow(X-mu, 2).sum(2, keepdim=False) / (2*torch.pow(self.sigma, 2))) )
        # Custom Function
        # Phi = torch.exp(-self.sigma.mul((X-mu).pow(2).sum(2, keepdim=False).sqrt() ) )
        # Multi-Quadric RBF
        Phi = torch.sqrt(self.sigma.pow(2).add((X - mu).pow(2).sum(2, keepdim=False)))
        # print((1e-9 + Phi.sum(dim=-1).unsqueeze(1)).size())
        # print("Phi", Phi.size())
        # Phi = Phi.divide( (1e-9 + Phi.sum(dim=-1).squeeze(0)) )

        #
        return Phi


class RBFNet3(nn.Module):
    def __init__(self, device):
        super(RBFNet3, self).__init__()
        self.centers = np.random.rand(128, 64)
        self.rbf = RBFLayer3(64, len(self.centers), self.centers, True, device)
        self.linear = nn.Linear(len(self.centers), 1)

    def forward(self, x):
        x = self.rbf(x)
        x = self.linear(x)
        x = torch.tanh(x)
        return x
