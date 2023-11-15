import pandas as pd
import torch
from torch.utils.data import Dataset

from src.lib.utilities import fen_to_tensor_one_board


#########################################################################
# This class is a dataset which reads .csv files with
# (FEN) positions and an evaluation.
#########################################################################
def to_tensor(fen_position):
    return fen_to_tensor_one_board(fen_position)


class PositionToEvaluationDataset(Dataset):
    def __init__(self, csv_file, device):
        self.data = pd.read_csv(csv_file)
        self.device = device

        # Calculate min and max evaluation scores for normalization
        self.min_score = self.data['Evaluation'].min()
        self.max_score = self.data['Evaluation'].max()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_fen = self.data.iloc[idx, 0]  # FEN position

        x = to_tensor(x_fen)  # Convert string to tensor
        y = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)  # position evaluation
        y_normalized = (y - self.min_score) / (self.max_score - self.min_score)  # Normalize evaluation
        return x.to(self.device), y_normalized.to(self.device)
