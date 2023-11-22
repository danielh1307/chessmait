import argparse
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from src.lib.utilities import fen_to_tensor_one_board
from src.model.ChessmaitMlp4 import ChessmaitMlp4

# Helper script for different actions in the context of regression models.
# See the documentation of the arguments for more information.

# Adjust these values for your needs
# The MAX_EVALUATION and MIN_EVALUATION can be taken from wandb
model = ChessmaitMlp4()
MAX_EVALUATION = 12352
MIN_EVALUATION = -12349
MODEL_NAME = "divine-leaf-29"

CLASSES = {
    ">4": {
        "max": 400
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
        "min": -400,
    }
}


def get_device():
    """
    Checks which device is most appropriate to perform the training.
    If cuda is available, cuda is returned, otherwise mps or cpu.

    Returns
    -------
    str
        the device which is used to perform the training.

    """
    _device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"For this training, we are going to use {_device} device ...")
    return _device


def fen_to_tensor(fen):
    return fen_to_tensor_one_board(fen)


def reverse_normalization(normalized_evaluation):
    return normalized_evaluation * (MAX_EVALUATION - MIN_EVALUATION) + MIN_EVALUATION


def evaluate_fen_by_model(fen_list, device):
    batch_tensors = [fen_to_tensor(fen).to(device) for fen in fen_list]

    with torch.no_grad():
        input_batch = torch.stack(batch_tensors)
        return model(input_batch)


def evaluate_fen_command_line(device):
    while True:
        fen = input("What is your FEN?")
        if fen == "q":
            break

        output = evaluate_fen_by_model([fen], device)

        print(f"My evaluation is: {output}")
        print(f"My evaluation normalized is: {reverse_normalization(output.item())}")


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
        f" ...done, it took me {end_time - start_time}s to do this and I now have {len(all_results)} evaluated results (normalized)")
    df['Evaluation_Predicted_Normalized'] = [result[0] for result in all_results]

    print(f"Un-normalizing the results ...")
    df['Evaluation_Predicted'] = df['Evaluation_Predicted_Normalized'].apply(reverse_normalization)

    output_file_name = f"{fen_file[0:-4]}_evaluated_{MODEL_NAME}.csv"
    print(f"Finished, saving the result to {output_file_name} ...")
    df.to_csv(output_file_name, index=False)


def evaluation_to_class(evaluation):
    for key, range_dict in CLASSES.items():
        if "min" in range_dict and "max" in range_dict:
            min_value = range_dict["min"]
            max_value = range_dict["max"]
            if min_value <= evaluation <= max_value:
                return key
        elif "min" in range_dict:
            min_value = range_dict["min"]
            if evaluation < min_value:
                return key
        elif "max" in range_dict:
            max_value = range_dict["max"]
            if evaluation >= max_value:
                return key
    raise Exception(f"No class found for {evaluation}")


def add_classes(fen_file):
    df = pd.read_csv(fen_file)

    # get class labels
    df["Evaluated_Class"] = df["Evaluation"].apply(evaluation_to_class)

    output_file_name = f"{fen_file[0:-4]}_with_class.csv"
    print(f"Finished, saving the result to {output_file_name} ...")
    df.to_csv(output_file_name, index=False)


def write_values_in_bars(curr_plot):
    for p in curr_plot.patches:
        curr_plot.annotate(format(p.get_height(), '.1f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           xytext=(0, 9),
                           textcoords='offset points')


def create_statistics(fen_directory_evaluated):
    dfs = []
    for fen_file_evaluated in os.listdir(fen_directory_evaluated):
        if not fen_file_evaluated.endswith(".csv"):
            continue
        print(f"Reading file {fen_file_evaluated} ...")
        _df = pd.read_csv(os.path.join(fen_directory_evaluated, fen_file_evaluated))
        if _df["Evaluation"].dtype == 'object':
            # filter the mates
            _df = _df[~_df['Evaluation'].str.startswith('#')]
        _df["Evaluation"] = _df["Evaluation"].astype(int)
        _df["Evaluation_Predicted"] = _df["Evaluation_Predicted"].astype(int)
        dfs.append(_df)

    df = pd.concat(dfs, ignore_index=True)

    print(f"Creating statistics for {fen_directory_evaluated}, which are {len(df)} positions ...")

    # add class labels
    df["Evaluated_Class"] = df["Evaluation"].apply(evaluation_to_class)
    df["Evaluated_Class_Predicted"] = df["Evaluation_Predicted"].apply(evaluation_to_class)

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
    # Plot: show distribution of predicted classes (absolute)
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
    class_avg_errors = df.groupby('Evaluated_Class')[['Evaluation', 'Evaluation_Predicted']].apply(
        lambda x: (x['Evaluation'] - x['Evaluation_Predicted']).mean()).reset_index()

    curr_plot = sns.barplot(x='Evaluated_Class', y=0, data=class_avg_errors, ax=plot_axes)
    plot_axes.set_title('Mean predicted error per class')
    plot_axes.set_xlabel('Class Label')
    plot_axes.set_ylabel('Mean error')
    write_values_in_bars(curr_plot)

    ##########################################################################
    # Plot: for each true label, show the predictions of the model
    ##########################################################################
    class_labels = list(CLASSES.keys())
    curr_x = 3
    curr_y = 0
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
    plot_axes = axes[8, 0]
    cm = confusion_matrix(df["Evaluated_Class"], df["Evaluated_Class_Predicted"], labels=class_labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=plot_axes, xticklabels=class_labels,
                yticklabels=class_labels)
    plot_axes.set_title("Confusion Matrix")
    plot_axes.set_xlabel("Predicted Classes")
    plot_axes.set_ylabel("True Classes")

    ##########################################################################
    # Plot: scatterplot true values vs. predicted values
    ##########################################################################
    scatter_axes = axes[8, 1]
    scatter_axes.scatter(df["Evaluation"], df["Evaluation_Predicted"], c='b', label='Predicted vs. True')
    scatter_axes.set_title("Predicted vs True Regression Values")
    scatter_axes.set_xlabel("Predicted Values")
    scatter_axes.set_ylabel("True Values")

    plt.tight_layout()

    plt.savefig('plot.png', bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chessmait regression evaluator")

    # if set to true, we get a command line, and we can evaluate single FEN positions
    parser.add_argument("--fen-evaluation", action='store_true', required=False)

    # we can predict all the values in the given file
    parser.add_argument("--fen-evaluation-file", type=str, required=False)

    # we can create statistics based on a predicted .csv file
    parser.add_argument("--statistics", type=str, required=False)

    # we can add class labels to a predicted .csv file
    parser.add_argument("--add-classes", type=str, required=False)
    args = parser.parse_args()

    device = get_device()
    model.to(device)
    model.load_state_dict(torch.load(f"models/{MODEL_NAME}.pth"))
    model.eval()

    if args.fen_evaluation:
        evaluate_fen_command_line(device)
    elif args.statistics:
        create_statistics(args.statistics)
    elif args.add_classes:
        add_classes(args.add_classes)
    elif args.fen_evaluation_file:
        evaluate_fen_file(args.fen_evaluation_file, device)
