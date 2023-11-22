import fnmatch
import os
import time

import pandas as pd

from src.lib.utilities import fen_to_cnn_tensor_alternative

# Helper script to create a pickle dataframe with the tensors
# already created, since that takes a lot of time

PATH_TO_DATAFILE = os.path.join("data", "preprocessed-classification")
PATH_TO_PICKLEFILE = os.path.join("data", "pickle-classification-fen-to-cnn-tensor-alternative")


def to_tensor(fen_position):
    return fen_to_cnn_tensor_alternative(fen_position)


matching_files = [file for file in os.listdir(PATH_TO_DATAFILE) if
                  fnmatch.fnmatch(file, "lichess_db_standard_rated_2023-09.1.1.csv")]
file_names = [os.path.basename(file) for file in matching_files]
for file_name in file_names:
    df = pd.read_csv(os.path.join(PATH_TO_DATAFILE, file_name))

    if df["Evaluation"].dtype == 'object':
        df = df[~df['Evaluation'].str.startswith('#')]
        df["Evaluation"] = df["Evaluation"].astype(int)

    print("Converting FEN to tensor ...")
    start_time = time.time()
    df["FEN"] = df["FEN"].apply(to_tensor)
    end_time = time.time()
    print(f"Done, it took me {end_time - start_time}s to do so ...")

    pickle_file = f"{PATH_TO_PICKLEFILE}/{file_name[0:-4]}.pkl"
    df.to_pickle(pickle_file)
    print(f"Pickle file {pickle_file} written ...")

print("Finished")
