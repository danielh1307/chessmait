import argparse
import math
import os
import random
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from src.lib.analytics_utilities import evaluation_to_class, remove_mates
from src.lib.utilities import fen_to_tensor_one_board, write_values_in_bars
from src.lib.utilities import get_device
from src.model.ChessmaitMlp5 import ChessmaitMlp5

# Helper script for different actions in the context of regression models.
# See the documentation of the arguments for more information.

# Example usages:
# Let the model predict all positions in the given CSV-FILE
# the CSV-FILE is expected to contain a FEN position and a ground truth evaluation
# $ python chessmait_regression_evaluation.py --fen-evaluation-file=<CSV-FILE>

# Create statistics which compare the ground truth to the prediction
# DIR is a directory (based on project root) which contains .csv files. The .csv files
# contain both the ground truth (Stockfish evaluation, column Evaluation)
# and the prediction from the model (column Evaluation_Predicted)
# $ python chessmait_regression_evaluation.py --statistics=<DIR>

# Adjust these values for your needs
# The MAX_EVALUATION and MIN_EVALUATION can be taken from wandb
model = ChessmaitMlp5()
NORMALIZATION_USED = False
MAX_EVALUATION = 15265
MIN_EVALUATION = -15265
CLIPPING_USED = True
MAX_CLIPPING = 1000
MIN_CLIPPING = -1000
MODEL_NAME = "lemon-plasma-103"

CLASSES = {
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


def fen_to_tensor(fen):
    return fen_to_tensor_one_board(fen)


def reverse_normalization(normalized_evaluation):
    return normalized_evaluation * (MAX_EVALUATION - MIN_EVALUATION) + MIN_EVALUATION


def evaluate_fen_by_model(fen_list, device):
    batch_tensors = [fen_to_tensor(fen).to(device) for fen in fen_list]

    with torch.no_grad():
        input_batch = torch.stack(batch_tensors)
        return model(input_batch)


def evaluate_fen_file(fen_file, device):
    # init some values
    batch_size = 5000
    current_idx = 0

    print(f"Loading data from {fen_file} into a dataframe ...")
    df = pd.read_csv(fen_file)
    print(f"Loaded {df.shape[0]} positions ...")
    print(f"Evaluating all positions in batches of size {batch_size}...")
    start_time = time.time()
    all_results = []
    while True:
        start_idx = current_idx
        if current_idx + batch_size < len(df):
            end_idx = current_idx + batch_size
        else:
            end_idx = len(df)

        # Get a batch of FEN values as a list
        fen_batch = df['FEN'].iloc[start_idx:end_idx].tolist()

        # Call evaluate_fen_by_model with the batch of FEN values
        results = evaluate_fen_by_model(fen_batch, device)

        all_results.extend(results.tolist())

        if end_idx == len(df):
            break
        current_idx = end_idx

    end_time = time.time()
    print(
        f" ...done, it took me {end_time - start_time}s to do this and I now have {len(all_results)} predicted results")
    df['Evaluation_Predicted_Original'] = [result[0] for result in all_results]

    if NORMALIZATION_USED:
        print(f"Un-normalizing the results ...")
        df['Evaluation_Predicted'] = df['Evaluation_Predicted_Original'].apply(reverse_normalization)

    if CLIPPING_USED:
        # we "clip" both the true values and the predicted values
        df = remove_mates(df, 'Evaluation')
        df["Evaluation"] = df["Evaluation"].clip(lower=MIN_CLIPPING, upper=MAX_CLIPPING)
        df["Evaluation_Predicted"] = df["Evaluation_Predicted_Original"].clip(lower=MIN_CLIPPING, upper=MAX_CLIPPING)

    output_file_name = f"{fen_file[0:-4]}_evaluated_{MODEL_NAME}.csv"
    print(f"Finished, saving the result to {output_file_name} ...")
    df.to_csv(output_file_name, index=False)


def smallest_and_highest_differences(fen_directory_evaluated):
    dfs = []
    for fen_file_evaluated in os.listdir(fen_directory_evaluated):
        if not fen_file_evaluated.endswith(".csv"):
            continue
        print(f"Reading file {fen_file_evaluated} ...")
        _df = pd.read_csv(os.path.join(fen_directory_evaluated, fen_file_evaluated))
        _df = remove_mates(_df, 'Evaluation')
        _df["Evaluation_Predicted"] = _df["Evaluation_Predicted"].astype(int)
        dfs.append(_df)

    df = pd.concat(dfs, ignore_index=True)

    print(f"Creating statistics for {fen_directory_evaluated}, which are {len(df)} positions ...")

    # remove clipped values
    if CLIPPING_USED:
        # do not check values which have been clipped because the difference is always 0
        df = df[(df['Evaluation'] > MIN_CLIPPING) & (df['Evaluation'] < MAX_CLIPPING)]

    # add class labels
    df["Evaluated_Class"] = df["Evaluation"].apply(lambda x: evaluation_to_class(CLASSES, x))
    df["Evaluated_Class_Predicted"] = df["Evaluation_Predicted"].apply(lambda x: evaluation_to_class(CLASSES, x))

    # add absolute difference
    df["Diff"] = abs(df["Evaluation"] - df["Evaluation_Predicted"])

    # get best and worse evaluations
    best_evaluations = df.nsmallest(10, "Diff")
    worse_evaluations = df.nlargest(10, "Diff")

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(best_evaluations.to_string(index=False))
    print(worse_evaluations.to_string(index=False))


def create_statistics(fen_directory_evaluated):
    dfs = []
    for fen_file_evaluated in os.listdir(fen_directory_evaluated):
        if not fen_file_evaluated.endswith(".csv"):
            continue
        print(f"Reading file {fen_file_evaluated} ...")
        _df = pd.read_csv(os.path.join(fen_directory_evaluated, fen_file_evaluated))
        _df = remove_mates(_df, 'Evaluation')
        _df["Evaluation_Predicted"] = _df["Evaluation_Predicted"].astype(int)
        dfs.append(_df)

    df = pd.concat(dfs, ignore_index=True)

    print(f"Creating statistics for {fen_directory_evaluated}, which are {len(df)} positions ...")

    # add class labels
    df["Evaluated_Class"] = df["Evaluation"].apply(lambda x: evaluation_to_class(CLASSES, x))
    df["Evaluated_Class_Predicted"] = df["Evaluation_Predicted"].apply(lambda x: evaluation_to_class(CLASSES, x))

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(9, 2, figsize=(36, 24))

    ##########################################################################
    # Plot: show distribution of true classes (absolute)
    ##########################################################################
    plot_axes = axes[0, 0]
    curr_plot = sns.countplot(x='Evaluated_Class', data=df, ax=plot_axes)
    plot_axes.set_title('Distribution of True Classes')
    plot_axes.set_xlabel('Class Label')
    plot_axes.set_ylabel('Frequency')
    write_values_in_bars(curr_plot)

    ##########################################################################
    # Plot: show distribution of predicted classes (absolute)
    ##########################################################################
    plot_axes = axes[0, 1]
    curr_plot = sns.countplot(x='Evaluated_Class_Predicted', data=df, ax=plot_axes)
    plot_axes.set_title('Distribution of Predicted Classes')
    plot_axes.set_xlabel('Class Label')
    plot_axes.set_ylabel('Frequency')
    write_values_in_bars(curr_plot)

    ##########################################################################
    # Plot: show distribution of true classes (percentage)
    ##########################################################################
    plot_axes = axes[1, 0]
    percentage_values = (df['Evaluated_Class'].value_counts() / len(df)) * 100
    curr_plot = sns.barplot(x=percentage_values.index, y=percentage_values.values, ax=plot_axes)
    plot_axes.set_title('Distribution of True Classes (%)')
    plot_axes.set_xlabel('Class Label')
    plot_axes.set_ylabel('Frequency')
    write_values_in_bars(curr_plot)

    ##########################################################################
    # Plot: show distribution of predicted classes (percentage)
    ##########################################################################
    plot_axes = axes[1, 1]
    percentage_values = (df['Evaluated_Class_Predicted'].value_counts() / len(df)) * 100
    curr_plot = sns.barplot(x=percentage_values.index, y=percentage_values.values, ax=plot_axes)
    plot_axes.set_title('Distribution of Predicted Classes (%)')
    plot_axes.set_xlabel('Class Label')
    plot_axes.set_ylabel('Frequency')
    write_values_in_bars(curr_plot)

    # ##########################################################################
    # Plot: show mean error per class label
    # ##########################################################################
    plot_axes = axes[2, 0]
    mean_error = (df['Evaluation'] - df['Evaluation_Predicted']).abs().mean()
    rounded_mean_error = math.ceil(mean_error)
    class_avg_errors = df.groupby('Evaluated_Class')[['Evaluation', 'Evaluation_Predicted']].apply(
        lambda x: (x['Evaluation'] - x['Evaluation_Predicted']).mean()).reset_index()

    curr_plot = sns.barplot(x='Evaluated_Class', y=0, data=class_avg_errors, ax=plot_axes)
    plot_axes.set_title(f'Mean predicted error per class (overall: {rounded_mean_error})')
    plot_axes.set_xlabel('Class Label')
    plot_axes.set_ylabel('Mean error')
    write_values_in_bars(curr_plot)

    ##########################################################################
    # Plot: for each true label, show the predictions of the model
    ##########################################################################
    class_labels = list(CLASSES.keys())
    curr_x = 2
    curr_y = 1
    for class_label in class_labels:
        plot_axes = axes[curr_x, curr_y]
        curr_x = curr_x if curr_y == 0 else curr_x + 1
        curr_y = 1 if curr_y == 0 else 0

        curr_plot = sns.countplot(x='Evaluated_Class_Predicted', data=df[df["Evaluated_Class"] == class_label],
                                  ax=plot_axes)
        plot_axes.set_title(f'Distribution of Predicted Classes for True Class {class_label}')
        plot_axes.set_xlabel('Class Label')
        plot_axes.set_ylabel('Frequency')
        write_values_in_bars(curr_plot)

    ##########################################################################
    # Plot: show a confusion matrix
    ##########################################################################
    plot_axes = axes[7, 1]
    cm = confusion_matrix(df["Evaluated_Class"], df["Evaluated_Class_Predicted"], labels=class_labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=plot_axes, xticklabels=class_labels,
                yticklabels=class_labels)
    plot_axes.set_title("Confusion Matrix")
    plot_axes.set_xlabel("Predicted Classes")
    plot_axes.set_ylabel("True Classes")

    ##########################################################################
    # Plot: scatterplot true values vs. predicted values
    ##########################################################################
    # we just show 0,1% of the data (randomly chosen), since otherwise the plot
    # is too full of data points
    num_points_to_display = int(0.001 * len(df))

    # Randomly select a subset of data points
    random_indices = random.sample(range(len(df)), num_points_to_display)

    # Extract the selected data points
    selected_data = df.iloc[random_indices]

    scatter_axes = axes[8, 0]
    scatter_axes.scatter(selected_data["Evaluation_Predicted"], selected_data["Evaluation"], c='b',
                         label='Predicted vs. True', marker='o', s=20, alpha=0.5)
    scatter_axes.set_title("Predicted vs True Regression Values (just 0,1%)")
    scatter_axes.set_xlabel("Predicted Values")
    scatter_axes.set_ylabel("True Values")

    # Group by the true values and calculate the mean of predicted values for each group
    grouped = df.groupby('Evaluation')['Evaluation_Predicted'].mean().reset_index()

    # Create the line plot
    plot_axes = axes[8, 1]
    plot_axes.plot(grouped['Evaluation'], grouped['Evaluation_Predicted'])
    plot_axes.set_xlabel('True Values')
    plot_axes.set_ylabel('Mean Predicted Values')
    plot_axes.set_title('Mean Predicted Values vs. True Values')

    plt.tight_layout()

    plt.savefig('plot.png', bbox_inches='tight')


if __name__ == "__main__":
    if CLIPPING_USED and NORMALIZATION_USED:
        raise Exception("Don't use clipping and normalization at the same time ...")

    parser = argparse.ArgumentParser(description="Chessmait regression evaluator")

    # we can predict all the values in the given file
    parser.add_argument("--fen-evaluation-file", type=str, required=False)

    # we can create statistics based on a predicted .csv file
    parser.add_argument("--statistics", type=str, required=False)

    # get the best and worse evaluations
    parser.add_argument("--best-worse", type=str, required=False)

    args = parser.parse_args()

    device = get_device()
    model.to(device)
    model.load_state_dict(torch.load(f"models/{MODEL_NAME}.pth", map_location=device))
    model.eval()

    if args.statistics:
        create_statistics(args.statistics)
    elif args.fen_evaluation_file:
        evaluate_fen_file(args.fen_evaluation_file, device)
    elif args.best_worse:
        smallest_and_highest_differences(args.best_worse)
