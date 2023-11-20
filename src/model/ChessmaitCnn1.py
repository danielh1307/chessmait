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
            nn.Linear(72 * 8 * 8, 1152),
            nn.ReLU(),
            nn.Linear(1152,144),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(144, 1)
        )

    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
