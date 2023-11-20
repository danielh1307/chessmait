import torch.nn as nn


# Network is based on the MLP bitmap architecture described
# in https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf
# Compared to ChessmaitMlp1 it has a lot of more neurons because it looks
# like in ChessmaitMlp1 we have a bias on our data.
class ChessmaitMlp4(nn.Module):
    def __init__(self):
        super(ChessmaitMlp4, self).__init__()

        # Define the input layer
        self.layer1 = nn.Sequential(
            nn.Linear(768, 2048),
            nn.ELU(),
            nn.BatchNorm1d(2048)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ELU(),
            nn.BatchNorm1d(2048)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ELU(),
            nn.BatchNorm1d(2048)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(2048, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output_layer(x)

        return x
