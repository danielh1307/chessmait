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
        super(ChessmaitCnn5, self).__init__()
        self.conv1 = nn.Conv2d(12, 20, kernel_size=5, padding=1)
        # output is [20, 6, 6]
        self.conv2 = nn.Conv2d(20, 50, kernel_size=3, padding=1)
        # output is [50, 6, 6]
        self.fc1 = nn.Linear(50 * 6 * 6, 500)
        self.fc2 = nn.Linear(500, 3)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(-1, 50*6*6)  # Flatten the output from the CNN layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # apply softmax
        return torch.softmax(x, dim=1)
