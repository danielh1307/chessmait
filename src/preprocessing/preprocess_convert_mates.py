import os.path
import re

import pandas as pd

from src.lib.utilities import get_files_from_pattern


# In recent trainings, we have removed the mates from the evaluation since they are no numbers.
# With this script, we transform the mates to numbers to include them in further training.

DATA_DIRECTORY = os.path.join("data", "angelo")
FILE_PATTERN = "lichess_db_standard_rated_2023-03-006.csv"

def convert_mate():
    file_names = get_files_from_pattern(DATA_DIRECTORY, FILE_PATTERN)
    for data_file in file_names:
        print(f"Reading file {data_file}")
        df = pd.read_csv(data_file)

        # first, we clip the values (we define a max and a min) for all integer values in the dataframe
        df['Evaluation'] = df.apply(lambda row: clip_values(row), axis=1)

        # next, we transform the mate
        df['Evaluation'] = df.apply(lambda row: checkmate(row), axis=1)

        # now we transform all future mates
        df['Evaluation'] = df.apply(lambda row: future_mates(row), axis=1)

        output_file_name = f"{data_file[0:-4]}_with_mate.csv"
        print(f"Finished, saving the result to {output_file_name} ...")
        df.to_csv(output_file_name, index=False)


def clip_values(row):
    eval_str = row["Evaluation"]
    if not represents_int(eval_str):
        return eval_str
    eval_num = int(eval_str)
    if eval_num > 1500:
        return '1500'
    elif eval_num < -1500:
        return '-1500'
    else:
        return str(eval_num)


def represents_int(s):
    try:
        int(s)
    except ValueError:
        return False
    else:
        return True


def checkmate(row):
    eval_str = row["Evaluation"]
    if eval_str == '##':
        fen = row['FEN']
        color_to_move = re.search(r'\s(b|w)\s', fen).group(1)
        if color_to_move == 'w':
            return '-2000'
        else:
            return '2000'
    return eval_str


def future_mates(row):
    eval_str = row['Evaluation']
    if represents_int(eval_str):
        return eval_str
    if eval_str == '#3':
        return '1700'
    elif eval_str == '#2':
        return '1800'
    elif eval_str == '#1':
        return '1900'
    elif eval_str == '#-3':
        return '-1700'
    elif eval_str == '#-2':
        return '-1800'
    elif eval_str == '#-1':
        return '-1900'
    elif eval_str.startswith('#-'):
        return '-1600'
    elif eval_str.startswith('#'):
        return '1600'
    else:
        return eval_str


if __name__ == "__main__":
    print("Converting mates ...")
    convert_mate()
