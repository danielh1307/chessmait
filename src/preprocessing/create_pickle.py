import os
import time

import pandas as pd

from src.lib.utilities import get_files_from_pattern
from src.lib.fen_to_tensor import fen_to_bitboard
from src.lib.analytics_utilities import remove_mates

# Helper script to create a pickle dataframe with the tensors
# already created, since that takes a lot of time

PATH_TO_DATAFILE = os.path.join("data", "angelo")
PATH_TO_PICKLEFILE = os.path.join("data")


def to_tensor(fen_position):
    return fen_to_bitboard(fen_position)


file_names = get_files_from_pattern(PATH_TO_DATAFILE, "*_with_mate.csv")
for file_name in file_names:
    df = pd.read_csv(file_name)
    df = remove_mates(df, "Evaluation")

    print("Converting FEN to tensor ...")
    start_time = time.time()
    df["FEN"] = df["FEN"].apply(to_tensor)
    end_time = time.time()
    print(f"Done, it took me {end_time - start_time}s to do so ...")

    pickle_file = f"{PATH_TO_PICKLEFILE}/{os.path.basename(file_name)[0:-4]}_bitboard.pkl"
    df.to_pickle(pickle_file)
    print(f"Pickle file {pickle_file} written ...")

print("Finished")
