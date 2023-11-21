import argparse
import fnmatch
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PATH_TO_DATAFILE = os.path.join("data", "preprocessed-classification")


def write_values_in_bars(curr_plot):
    for p in curr_plot.patches:
        curr_plot.annotate(format(p.get_height(), '.1f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           xytext=(0, 9),
                           textcoords='offset points')


def analyze_files(file_pattern):
    matching_files = [file for file in os.listdir(PATH_TO_DATAFILE) if
                      fnmatch.fnmatch(file, file_pattern)]
    file_names = [os.path.basename(file) for file in matching_files]

    _dataframes = []
    for file_name in file_names:
        print(f"Read {file_names} ...")
        _dataframes.append(pd.read_csv(os.path.join(PATH_TO_DATAFILE, file_name)))

    df = pd.concat(_dataframes, ignore_index=True)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create classification files from regression files")

    parser.add_argument("--file-pattern", type=str, required=True)
    args = parser.parse_args()
    analyze_files(args.file_pattern)