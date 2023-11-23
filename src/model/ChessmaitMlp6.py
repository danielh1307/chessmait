import torch.nn as nn


# Network is based on the MLP bitmap architecture described
# in https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf
# This model has more neurons than ChessmaitMlp1, but it is more similar to it
# than ChessmaitMlp4. It also uses ReLU and a dropout instead of Batch Normalization.
# The batch normalization does not work well with Adam, see — for example, this run
# here: https://wandb.ai/chessmait/chessmait/runs/bmdm0m4t?workspace=user-hamm-daniel
class ChessmaitMlp6(nn.Module):
    def __init__(self):
        """
        Initializes a multi-layer perceptron (MLP) model for a regression task.

        This model consists of four sequential layers:
        - Layer 1: Input size 768, output size 2048, ReLU activation, and 20% dropout.
        - Layer 2: Input size 2048, output size 2048, ReLU activation, and 20% dropout.
        - Layer 3: Input size 2048, output size 2048, ReLU activation, and 20% dropout.
        - Output Layer: Input size 2048, output size 1.

        The forward pass applies these layers sequentially to the input tensor.

        Args:
            None

        Returns:
            None
        """
        super(ChessmaitMlp6, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(768, 1)
        )

    def forward(self, x):
        """
        Forward pass of the MLP model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 768).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)
        return x
