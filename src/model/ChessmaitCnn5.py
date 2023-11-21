import torch
import torch.nn as nn
import torch.nn.functional as F


# Convolutional Network
# It accepts input data of shape [12, 8, 8] and performs two convolutions
# with different kernel sizes.
# This is followed by a fully connected layer of 500 units.
# Dropout of 0.3 is used after the convolutional layers and the fully connected layer.
# It is created for classification problems (three classes).
class ChessmaitCnn5(nn.Module):
    def __init__(self):
        """
        Initializes a convolutional neural network (CNN) classification model.

        This model consists of two convolutional layers followed by fully connected layers for classification:
        - Convolutional Layer 1: Input channels 12, output channels 20, kernel size 5x5, and padding 1.
        - Convolutional Layer 2: Input channels 20, output channels 50, kernel size 3x3, and padding 1.
        - Fully Connected Layer 1: Input size 50*6*6, output size 500.
        - Fully Connected Layer 2: Input size 500, output size 3 (for classification).
        - Dropout with a rate of 30% applied after each ReLU activation.

        The forward pass applies these layers sequentially to the input tensor and applies softmax activation for classification.

        Args:
            None

        Returns:
            None
        """
        super(ChessmaitCnn5, self).__init__()
        self.conv1 = nn.Conv2d(12, 20, kernel_size=5, padding=1)
        # output is [20, 6, 6]
        self.conv2 = nn.Conv2d(20, 50, kernel_size=3, padding=1)
        # output is [50, 6, 6]
        self.fc1 = nn.Linear(50 * 6 * 6, 500)
        self.fc2 = nn.Linear(500, 3)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Forward pass of the CNN classification model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 12, 6, 6).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 3) with softmax activation for classification.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(-1, 50 * 6 * 6)  # Flatten the output from the CNN layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # apply softmax
        return torch.softmax(x, dim=1)
