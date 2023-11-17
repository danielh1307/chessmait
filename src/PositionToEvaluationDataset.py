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
    def __init__(self, csv_files, device):
        print("Loading the data ...")

        _dataframes = [pd.read_csv(csv_file) for csv_file in csv_files]
        self.data = pd.concat(_dataframes, ignore_index=True)

        # Calculate min and max evaluation scores for normalization
        self.min_score = self.data["Evaluation"].min()
        self.max_score = self.data["Evaluation"].max()

        # Normalize the evaluation in the range of [0, 1]
        print("Normalizing the evaluation ...")
        self.data["Evaluation"] = (self.data["Evaluation"] - self.min_score) / (self.max_score - self.min_score)

        print("Converting FEN to tensor ...")
        self.data["FEN"] = self.data["FEN"].apply(to_tensor)

    def __len__(self):
        return len(self.data)

    def get_min_max_score(self):
        return self.min_score, self.max_score

    def __getitem__(self, idx):
        x = self.data.iloc[idx, 0]  # FEN position
        y = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)  # position evaluation
        return x, y
