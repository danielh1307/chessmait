import argparse
import os

import pandas as pd

directory_preprocessed = os.path.join("data", "preprocessed")
directory_preprocessed_classification = os.path.join("data", "preprocessed-classification")

CLASSES = {
    "DRAW": {
        "label": 0
    },
    "WIN_WHITE": {
        "min": 150,
        "label": 1
    },
    "WIN_BLACK": {
        "max": -150,
        "label": 10
    }
}


def evaluation_to_class(evaluation):
    for key, range_dict in CLASSES.items():
        if "min" in range_dict and "max" in range_dict:
            min_value = range_dict["min"]
            max_value = range_dict["max"]
            if min_value <= evaluation <= max_value:
                return range_dict["label"]
        elif "min" in range_dict:
            min_value = range_dict["min"]
            if evaluation > min_value:
                return range_dict["label"]
        elif "max" in range_dict:
            max_value = range_dict["max"]
            if evaluation < max_value:
                return range_dict["label"]

    return 0


def preprocess_regression_to_evaluation(evaluated_regression_file):
    df = pd.read_csv(os.path.join(directory_preprocessed, evaluated_regression_file))
    if df["Evaluation"].dtype == 'object':
        # filter the mates
        df = df[~df['Evaluation'].str.startswith('#')]
        df["Evaluation"] = df["Evaluation"].astype(int)

    df["Evaluated_Class"] = df["Evaluation"].apply(evaluation_to_class)
    df = df.drop(columns=['Evaluation'])

    df.to_csv(os.path.join(directory_preprocessed_classification, evaluated_regression_file), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create classification files from regression files")

    parser.add_argument("--evaluated-regression-file", type=str, required=True)
    args = parser.parse_args()

    preprocess_regression_to_evaluation(args.evaluated_regression_file)
