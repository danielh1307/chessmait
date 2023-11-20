import torch.nn as nn


# Network is based on the MLP bitmap architecture described
# in https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf
# This model has more neurons than ChessmaitMlp1, but it is more similar to it
# than ChessmaitMlp4. It also uses ReLU and a dropout instead of Batch Normalization.
# The batch normalization does not work well with Adam, see — for example, this run
# here: https://wandb.ai/chessmait/chessmait/runs/bmdm0m4t?workspace=user-hamm-daniel
class ChessmaitMlp5(nn.Module):
    def __init__(self):
        super(ChessmaitMlp5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(768, 2048),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.2)
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
