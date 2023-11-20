import torch.nn as nn
import torch as torch


# Chessmait model using CNN
class ChessmaitCnn1(nn.Module):
    def __init__(self):
        super(ChessmaitCnn1,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12, 36, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(36, 72, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Dropout2d(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(72 * 6 * 6, 324),
            nn.ReLU(),
            nn.Linear(324, 54),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(54, 1)
        )

    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
