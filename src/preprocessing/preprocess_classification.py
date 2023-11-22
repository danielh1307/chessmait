import argparse
import os

import pandas as pd
from lib.analytics_utilities import evaluation_to_class

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
        "label": 2
    }
}


def preprocess_regression_to_evaluation(evaluated_regression_file):
    df = pd.read_csv(os.path.join(directory_preprocessed, evaluated_regression_file))
    if df["Evaluation"].dtype == 'object':
        # filter the mates
        df = df[~df['Evaluation'].str.startswith('#')]
        df["Evaluation"] = df["Evaluation"].astype(int)

    df["Evaluated_Class"] = df["Evaluation"].apply(lambda x: evaluation_to_class(CLASSES, x))
    df = df.drop(columns=['Evaluation'])

    out_file = os.path.join(directory_preprocessed_classification, evaluated_regression_file)
    df.to_csv(out_file, index=False)
    print(f"File written to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create classification files from regression files")

    parser.add_argument("--evaluated-regression-file", type=str, required=True)
    args = parser.parse_args()

    preprocess_regression_to_evaluation(args.evaluated_regression_file)
