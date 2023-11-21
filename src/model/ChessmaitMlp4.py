import torch.nn as nn


# Network is based on the MLP bitmap architecture described
# in https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf

# Compared to ChessmaitMlp1 it has a lot of more neurons because it looks
# like in ChessmaitMlp1 we have a bias on our data.
class ChessmaitMlp4(nn.Module):
    def __init__(self):
        """
        Initializes a multi-layer perceptron (MLP) model for a regression task.

        This model consists of four sequential layers with batch normalization:
        - Layer 1: Input size 768, output size 2048, ELU activation, and batch normalization.
        - Layer 2: Input size 2048, output size 2048, ELU activation, and batch normalization.
        - Layer 3: Input size 2048, output size 2048, ELU activation, and batch normalization.
        - Output Layer: Input size 2048, output size 1.

        The forward pass applies these layers sequentially to the input tensor.

        Args:
            None

        Returns:
            None
        """
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
        x = self.output_layer(x)

        return x
