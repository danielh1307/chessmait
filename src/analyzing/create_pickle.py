import os
import time

import pandas as pd

from src.lib.utilities import fen_to_tensor_one_board_dense, get_files_from_pattern
from src.lib.analytics_utilities import remove_mates

# Helper script to create a pickle dataframe with the tensors
# already created, since that takes a lot of time

PATH_TO_DATAFILE = os.path.join("data", "angelo")
PATH_TO_PICKLEFILE = os.path.join("data")


def to_tensor(fen_position):
    return fen_to_tensor_one_board_dense(fen_position)


file_names = get_files_from_pattern(PATH_TO_DATAFILE, "x_lichess_db_standard_rated_2023-03-001.csv")
for file_name in file_names:
    df = pd.read_csv(os.path.join(PATH_TO_DATAFILE, file_name))
    df = remove_mates(df, "Evaluation")

    print("Converting FEN to tensor ...")
    start_time = time.time()
    df["FEN"] = df["FEN"].apply(to_tensor)
    end_time = time.time()
    print(f"Done, it took me {end_time - start_time}s to do so ...")

    pickle_file = f"{PATH_TO_PICKLEFILE}/{file_name[0:-4]}.pkl"
    df.to_pickle(pickle_file)
    print(f"Pickle file {pickle_file} written ...")

print("Finished")
