import torch
from torch.utils.data import Dataset

from src.lib.analytics_utilities import remove_mates
from src.lib.utilities import fen_to_tensor_one_board, dataframe_from_files

USE_NORMALIZATION = True
USE_CLIPPING = False
MIN_CLIPPING = -1000
MAX_CLIPPING = 1000


#########################################################################
# This class is a dataset which reads .csv files with
# (FEN) positions and an evaluation.
#########################################################################
def to_tensor(fen_position):
    return fen_to_tensor_one_board(fen_position)


class PositionToEvaluationDataset(Dataset):
    def __init__(self, csv_files, pickle_files):
        if USE_NORMALIZATION and USE_CLIPPING:
            raise Exception("Either choose normalization or clipping, not both")

        print("Loading the data ...")
        load_csv = len(csv_files) > 0
        load_pickle = len(pickle_files) > 0

        if load_csv and load_pickle:
            raise Exception("Either choose .csv files or .pkl files to load")

        _dataframes = []
        if load_pickle:
            self.data = dataframe_from_files(pickle_files, pickle_files=True)

        if load_csv:
            self.data = dataframe_from_files(csv_files)
            self.data = remove_mates(self.data, "Evaluation")

        if USE_CLIPPING:
            print("Clip the values between -1000 and 1000")
            self.data["Evaluation"] = self.data["Evaluation"].clip(lower=MIN_CLIPPING, upper=MAX_CLIPPING)

        # Calculate min and max evaluation scores
        self.min_score = self.data["Evaluation"].min()
        self.max_score = self.data["Evaluation"].max()

        if USE_NORMALIZATION:
            print("Normalizing the evaluation ...")
            self.data["Evaluation"] = (self.data["Evaluation"] - self.min_score) / (self.max_score - self.min_score)

        if load_csv:
            print("Converting FEN to tensor ...")
            self.data["FEN"] = self.data["FEN"].apply(to_tensor)

        print("All data loaded ...")

    def __len__(self):
        return len(self.data)

    def get_min_max_score(self):
        return self.min_score, self.max_score

    def __getitem__(self, idx):
        x = self.data.iloc[idx, 0]  # FEN position
        y = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)  # position evaluation
        return x, y
