import torch.nn as nn


# Compared to the previous nn (ChessmaitMlp2.py), we have removed
# the Batch Normalization layers.
class ChessmaitMlp3(nn.Module):
    def __init__(self):
        super(ChessmaitMlp3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(768, 1048),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1048, 500),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(500, 50),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(50, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output_layer(x)
        return x
