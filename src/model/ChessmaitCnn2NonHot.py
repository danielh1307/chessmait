import torch.nn as nn
import torch as torch


# Chessmait model using CNN
class ChessmaitCnn2NonHot(nn.Module):
    def __init__(self):
        super(ChessmaitCnn2NonHot,self).__init__()
        self.conv1 = nn.Sequential(
            # Conv layer 1
            nn.Conv2d(2, 24, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Dropout2d()
        )

        self.conv2 = nn.Sequential(
            # Conv layer 2
            nn.Conv2d(24, 48, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Dropout2d()
        )

        self.conv3 = nn.Sequential(
            # Conv layer 3
            nn.Conv2d(48, 96, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Dropout2d()
        )

        self.lin1 = nn.Sequential(
            nn.Linear(96 * 8 * 8, 384),
            nn.ReLU(),
            nn.Dropout()
        )

        self.lin2 = nn.Sequential(
            nn.Linear(384, 48),
            nn.ReLU(),
            nn.Dropout()
        )

        self.classifier = nn.Sequential(
            nn.Linear(48, 1)
        )

    def forward(self,x):
        print("Forward")
        print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.classifier(x)
        return x
