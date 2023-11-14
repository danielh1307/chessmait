import torch.nn as nn


# Network is based on the MLP bitmap architecture described
# in https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf
# Compared to ChessmaitMlp1, we try to reduce overfitting here.
# What we are doing to reduce overfitting:
# - increase dropout from 0.2 to 0.3
# - added batch normalization layers
# - learning rate schedule (implemented in the training process, not the model)
class ChessmaitMlp2(nn.Module):
    def __init__(self):
        super(ChessmaitMlp2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(768, 1048),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.bn1 = nn.BatchNorm1d(1048)
        self.layer2 = nn.Sequential(
            nn.Linear(1048, 500),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.bn2 = nn.BatchNorm1d(500)
        self.layer3 = nn.Sequential(
            nn.Linear(500, 50),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.bn3 = nn.BatchNorm1d(50)
        self.output_layer = nn.Sequential(
            nn.Linear(50, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.bn1(self.layer1(x))
        x = self.bn2(self.layer2(x))
        x = self.bn3(self.layer3(x))
        x = self.output_layer(x)
        return x
