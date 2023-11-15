import pandas as pd
import torch
from torch.utils.data import Dataset

from src.lib.utilities import fen_to_tensor, CLASS_WIN, CLASS_DRAW, CLASS_LOSS


#########################################################################
# This class is a dataset which reads .csv files with
# (FEN) positions and an evaluation.
#########################################################################
def to_tensor(fen_position):
    return fen_to_tensor(fen_position)


def evaluation_to_class(y):
    if y > 150:
        return CLASS_WIN
    elif y < -150:
        return CLASS_LOSS
    else:
        return CLASS_DRAW


class PositionToEvaluationDatasetClassification(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_fen = self.data.iloc[idx, 0]  # FEN position

        x = to_tensor(x_fen)  # Convert string to tensor
        y = torch.tensor(evaluation_to_class(self.data.iloc[idx, 1]), dtype=torch.long)
        return x, y