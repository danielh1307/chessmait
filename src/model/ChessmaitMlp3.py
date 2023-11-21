import torch.nn as nn


# Compared to the previous nn (ChessmaitMlp2.py), we have removed
# the Batch Normalization layers.
class ChessmaitMlp3(nn.Module):
    def __init__(self):
        """
        Initializes a multi-layer perceptron (MLP) model for a regression task.

        This model consists of four sequential layers without batch normalization:
        - Layer 1: Input size 768, output size 1048, ReLU activation, and 30% dropout.
        - Layer 2: Input size 1048, output size 500, ReLU activation, and 30% dropout.
        - Layer 3: Input size 500, output size 50, ReLU activation, and 30% dropout.
        - Output Layer: Input size 50, output size 1.

        The forward pass applies these layers sequentially to the input tensor.

        Args:
            None

        Returns:
            None
        """
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
