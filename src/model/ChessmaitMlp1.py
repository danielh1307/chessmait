import torch
import torch.nn as nn


# Network is based on the MLP bitmap architecture described
# in https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf
class ChessmaitMlp1(nn.Module):
    def __init__(self):
        super(ChessmaitMlp1, self).__init__()
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
            nn.Linear(50, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output_layer(x)
        return x


# Create an instance of the model
model = ChessmaitMlp1()

# Define the Adam optimizer with specified parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.90, 0.99), eps=1e-8)

# Define the loss function for regression (e.g., Mean Squared Error)
criterion = nn.MSELoss()
