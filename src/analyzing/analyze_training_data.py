import argparse
import fnmatch
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.lib.analytics_utilities import evaluation_to_class, remove_mates

# Helper script to analyze training data, mainly based on classification
# It creates seaborn plots of the given .csv files.


PATH_TO_DATAFILE = os.path.join("data", "angelo")

REGRESSION_CLASSES = {
    ">4": {
        "min": 400
    },
    "4>p>2": {
        "min": 200,
        "max": 400
    },
    "2>p>1": {
        "min": 100,
        "max": 200
    },
    "1>p>.5": {
        "min": 50,
        "max": 100
    },
    ".5>p>0": {
        "min": 0,
        "max": 50
    },
    "0>p>-0.5": {
        "min": -50,
        "max": 0
    },
    "-0.5>p>-1": {
        "min": -100,
        "max": -50
    },
    "-1>p>-2": {
        "min": -200,
        "max": -100
    },
    "-2>p>-4": {
        "min": -400,
        "max": -200
    },
    "<-4": {
        "max": -400,
    }
}


def write_values_in_bars(curr_plot):
    for p in curr_plot.patches:
        curr_plot.annotate(format(p.get_height(), '.1f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           xytext=(0, 9),
                           textcoords='offset points')


def analyze_files_classification(file_pattern):
    matching_files = [file for file in os.listdir(PATH_TO_DATAFILE) if
                      fnmatch.fnmatch(file, file_pattern)]
    file_names = [os.path.basename(file) for file in matching_files]

    _dataframes = []
    for file_name in file_names:
        print(f"Read {file_names} ...")
        _dataframes.append(pd.read_csv(os.path.join(PATH_TO_DATAFILE, file_name)))

    df = pd.concat(_dataframes, ignore_index=True)
    print(f"I have loaded {len(df)} entries ...")

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))

    plot_axes = axes[0]
    curr_plot = sns.countplot(x='Evaluated_Class', data=df, ax=plot_axes)
    plot_axes.set_title('Distribution of True Classes')
    plot_axes.set_xlabel('Class Label')
    plot_axes.set_ylabel('Frequency')
    write_values_in_bars(curr_plot)

    plot_axes = axes[1]
    percentage_values = (df['Evaluated_Class'].value_counts() / len(df)) * 100
    curr_plot = sns.barplot(x=percentage_values.index, y=percentage_values.values, ax=plot_axes)
    plot_axes.set_title('Distribution of True Classes (%)')
    plot_axes.set_xlabel('Class Label')
    plot_axes.set_ylabel('Frequency')
    write_values_in_bars(curr_plot)

    plt.tight_layout()
    plt.show()


def analyze_files_regression(file_pattern):
    matching_files = [file for file in os.listdir(PATH_TO_DATAFILE) if
                      fnmatch.fnmatch(file, file_pattern)]
    file_names = [os.path.basename(file) for file in matching_files]

    _dataframes = []
    for file_name in file_names:
        print(f"Read {file_names} ...")
        _dataframes.append(pd.read_csv(os.path.join(PATH_TO_DATAFILE, file_name)))

    df = pd.concat(_dataframes, ignore_index=True)
    print(f"I have loaded {len(df)} entries ...")

    df = remove_mates(df, "Evaluation")
    # add class labels
    df["Evaluated_Class"] = df["Evaluation"].apply(lambda x: evaluation_to_class(REGRESSION_CLASSES, x))

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))

    ##########################################################################
    # Plot: show distribution of true classes (absolute)
    ##########################################################################
    plot_axes = axes[0]
    curr_plot = sns.countplot(x='Evaluated_Class', data=df, ax=plot_axes)
    plot_axes.set_title('Distribution of True Classes')
    plot_axes.set_xlabel('Class Label')
    plot_axes.set_ylabel('Frequency')
    write_values_in_bars(curr_plot)

    ##########################################################################
    # Plot: show distribution of true classes (percentage)
    ##########################################################################
    plot_axes = axes[1]
    percentage_values = (df['Evaluated_Class'].value_counts() / len(df)) * 100
    curr_plot = sns.barplot(x=percentage_values.index, y=percentage_values.values, ax=plot_axes)
    plot_axes.set_title('Distribution of True Classes (%)')
    plot_axes.set_xlabel('Class Label')
    plot_axes.set_ylabel('Frequency')
    write_values_in_bars(curr_plot)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze training data")

    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--file-pattern", type=str, required=True)
    args = parser.parse_args()
    if args.type == "regression":
        analyze_files_regression(args.file_pattern)
    elif args.type == "classification":
        analyze_files_classification(args.file_pattern)
    else:
        raise Exception("Please pass valid type (either regression or classification)")
