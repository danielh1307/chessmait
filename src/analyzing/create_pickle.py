import fnmatch
import os
import time

import pandas as pd

from src.lib.utilities import fen_to_tensor_one_board
PATH_TO_DATAFILE = os.path.join("data", "preprocessed-classification")
PATH_TO_PICKLEFILE = os.path.join("data", "pickle-classification-fen-to-tensor-one-board")


def to_tensor(fen_position):
    return fen_to_tensor_one_board(fen_position)


matching_files = [file for file in os.listdir(PATH_TO_DATAFILE) if
                  fnmatch.fnmatch(file, "kaggle_preprocessed_1000.csv")]
file_names = [os.path.basename(file) for file in matching_files]
for file_name in file_names:
    df = pd.read_csv(os.path.join(PATH_TO_DATAFILE, file_name))

    print("Converting FEN to tensor ...")
    start_time = time.time()
    df["FEN"] = df["FEN"].apply(to_tensor)
    end_time = time.time()
    print(f"Done, it took me {end_time - start_time}s to do so ...")

    df.to_pickle(f"{PATH_TO_PICKLEFILE}/{file_name[0:-4]}.pkl")
print("Finished")
