import pandas as pd
import torch
from torch.utils.data import Dataset

from src.lib.analytics_utilities import remove_mates
from src.lib.utilities import fen_to_cnn_tensor_alternative


#########################################################################
# This class is a dataset which reads .csv files with
# (FEN) positions and evaluated classes.
#########################################################################
def to_tensor(fen_position):
    return fen_to_cnn_tensor_alternative(fen_position)


class PositionToEvaluationDatasetClassification(Dataset):
    def __init__(self, csv_files, pickle_files):
        load_csv = len(csv_files) > 0
        load_pickle = len(pickle_files) > 0

        if load_csv and load_pickle:
            raise Exception("Either choose .csv files or .pkl files to load")

        _dataframes = []
        if load_pickle:
            for pickle_file in pickle_files:
                print("Loading ", pickle_file, " ...")
                _dataframe = pd.read_pickle(pickle_file)
                _dataframes.append(_dataframe)
        if load_csv:
            for csv_file in csv_files:
                print("Loading ", csv_file, " ...")
                _dataframe = pd.read_csv(csv_file)
                _dataframe = remove_mates(_dataframe, "Evaluation")
                _dataframes.append(_dataframe)

        self.data = pd.concat(_dataframes, ignore_index=True)

        if load_csv:
            print("Converting FEN to tensor ...")
            self.data["FEN"] = self.data["FEN"].apply(to_tensor)

        print("All data loaded ...")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data.iloc[idx, 0]  # FEN position
        y = torch.tensor(self.data.iloc[idx, 1], dtype=torch.long) # class position
        return x, y