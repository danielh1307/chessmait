import torch
import torch.nn as nn


# Network is based on the MLP bitmap architecture described
# in https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf
class ChessmaitMlp1Classification(nn.Module):
    """
    Initializes a multi-layer perceptron (MLP) classification model.

    This model consists of four sequential layers for classification:
    - Layer 1: Input size 768, output size 1048, ReLU activation, and 20% dropout.
    - Layer 2: Input size 1048, output size 500, ReLU activation, and 20% dropout.
    - Layer 3: Input size 500, output size 50, ReLU activation, and 20% dropout.
    - Output Layer: Input size 50, output size 3 (for classification).

    The forward pass applies these layers sequentially to the input tensor and applies softmax activation for classification.

    Args:
        None

    Returns:
        None
    """
    def __init__(self):
        super(ChessmaitMlp1Classification, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(768, 1048),
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
        self.output_layer = nn.Sequential(
            nn.Linear(50, 3)
        )

    def forward(self, x):
        """
        Forward pass of the MLP classification model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 768).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 3) with softmax activation for classification.
        """
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output_layer(x)
        return torch.softmax(x, dim=1)
