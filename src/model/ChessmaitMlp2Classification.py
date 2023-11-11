import torch.nn as nn


# Network is based on the MLP bitmap architecture described
# in https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf

# Here we are using a network for a classification problem instead of regression.
# We want to predict 3 classes: win, draw, loss.
class ChessmaitMlp2Classification(nn.Module):
    def __init__(self):
        super(ChessmaitMlp2Classification, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(384, 1048),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1048, 500),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(500, 50),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output_layer = nn.Linear(50, 3)  # output layer for 3 classes

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output_layer(x)
        return nn.functional.softmax(x, dim=1)  # softmax activation for a classification problem
