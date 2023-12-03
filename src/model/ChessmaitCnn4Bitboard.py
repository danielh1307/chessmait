import torch.nn as nn
import torch as torch


class ChessmaitCnn4Bitboard(nn.Module):

    def __init__(self):
        super(ChessmaitCnn4Bitboard,self).__init__()

        # input size = 8 (rows) x 8 (cols) x 16 (bitboards)
        # - 6 bitboards for white pieces
        # - 6 bitboards for black pieces
        # - 1 for empty squares
        # - 1 for castling rights
        # - 1 for en passant
        # - 1 for player

        self.features = nn.Sequential(
            # Conv layer 1
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Conv layer 2
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            # first fully connected layer 8192 => 8192
            nn.Linear(128 * 64,128 * 64),
            nn.ReLU(),

        # second fully connected layer 8192 => 4096
            nn.Linear(128 * 64,64 * 64),
            nn.ReLU(),

            nn.Linear(64 * 64,1),
        )

    def forward(self,x,debug=False):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x