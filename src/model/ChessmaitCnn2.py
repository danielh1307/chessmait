import torch.nn as nn
import torch as torch


# Chessmait model using CNN
class ChessmaitCnn2(nn.Module):
    def __init__(self):
        super(ChessmaitCnn2, self).__init__()
        self.features = nn.Sequential(
            # Conv layer 1
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),

            # Conv layer 2
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            # Conv layer 3
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(96 * 8 * 8, 384),
            nn.ReLU(),
            nn.Linear(384, 48),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(48, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
