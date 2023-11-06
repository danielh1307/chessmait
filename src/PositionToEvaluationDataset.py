import pandas as pd
import torch
from torch.utils.data import Dataset


#########################################################################
# This class is a dataset which reads .csv files with
# (FEN) positions and an evaluation.
#########################################################################
class PositionToEvaluationDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def to_tensor(self, string):
        # FIXME: transform FEN to tensor
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_fen = self.data.iloc[idx, 0]  # FEN position

        x = self.to_tensor(x_fen)  # Convert string to tensor
        y = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)  # position evaluation
        return x, y
