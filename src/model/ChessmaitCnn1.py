import torch.nn as nn
import torch as torch


# Chessmait model using CNN
class ChessmaitCnn1(nn.Module):
    def __init__(self):
        super(ChessmaitCnn1,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 8 * 8, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096,256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 1)
        )

    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
