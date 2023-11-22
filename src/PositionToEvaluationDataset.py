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
    def __init__(self, csv_files, pickle_files):
        print("Loading the data ...")
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
                # we have some .csv files with Evaluation as string, since the mate
                # is also contained there (e.g. 'mate in 3')
                # we remove those lines and convert the datatype to int
                _dataframe = pd.read_csv(csv_file)
                if _dataframe["Evaluation"].dtype == 'object':
                    # remove mates
                    _dataframe = _dataframe[~_dataframe['Evaluation'].str.startswith('#')]
                    _dataframe["Evaluation"] = _dataframe["Evaluation"].astype(int)
                _dataframes.append(_dataframe)

        self.data = pd.concat(_dataframes, ignore_index=True)

        # With this code you can clip the values between a specific range,
        # e.g. between (in pawns) +15 (1500) and -15 (-1500)
        # print("Clip the values between -1500 and 1500")
        # self.data["Evaluation"] = self.data["Evaluation"].clip(lower=-1500, upper=1500)

        # Calculate min and max evaluation scores for normalization
        self.min_score = self.data["Evaluation"].min()
        self.max_score = self.data["Evaluation"].max()

        # With this code you can normalize the evaluation in a specific range (e.g. of [0, 1])
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
