import torch.nn as nn


# Network base on ChessmaitMlp5 but with the adaption to the tensor fen_to_tensor_simple.
class ChessmaitMlp6Simple(nn.Module):
    def __init__(self):
        """
        Initializes a multi-layer perceptron (MLP) model for a regression task.

        This model consists of four sequential layers:
        - Layer 1: Input size 769, output size 1538, ReLU activation, and 20% dropout.
        - Layer 2: Input size 1538, output size 1538, ReLU activation, and 20% dropout.
        - Layer 3: Input size 1538, output size 1538, ReLU activation, and 20% dropout.
        - Output Layer: Input size 1538, output size 1.

        The forward pass applies these layers sequentially to the input tensor.

        Args:
            None

        Returns:
            None
        """
        super(ChessmaitMlp6Simple, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(8*8*6*2+1, 1538),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1538, 1538),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(1538, 1538),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(1538, 1)
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
