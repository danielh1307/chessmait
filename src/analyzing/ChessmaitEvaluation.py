import argparse
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from src.lib.utilities import fen_to_tensor_one_board
from src.model.ChessmaitMlp1 import ChessmaitMlp1

MAX_EVALUATION = 12352
MIN_EVALUATION = -12349
MODEL_NAME = "firm-star-24"


def fen_to_tensor(fen):
    return fen_to_tensor_one_board(fen)


def reverse_normalization(normalized_evaluation):
    return normalized_evaluation * (MAX_EVALUATION - MIN_EVALUATION) + MIN_EVALUATION


def evaluate_fen_by_model(fen_list):
    batch_tensors = [fen_to_tensor(fen).to("cuda") for fen in fen_list]

    with torch.no_grad():
        input_batch = torch.stack(batch_tensors)
        return model(input_batch)


def evaluate_fen_command_line():
    while True:
        fen = input("What is your FEN?")
        if fen == "q":
            break

        output = evaluate_fen_by_model([fen])

        print(f"My evaluation is: {output}")
        print(f"My evaluation normalized is: {reverse_normalization(output.item())}")


def evaluate_fen_file(fen_file):
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
        results = evaluate_fen_by_model(fen_batch)

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
    evaluation = int(evaluation)
    if evaluation >= 500:
        return ">5"
    elif 500 > evaluation >= 300:
        return "5>p>3"
    elif 300 > evaluation >= 100:
        return "3>p>1"
    elif 100 > evaluation >= -100:
        return "1>p>-1"
    elif -100 > evaluation >= -300:
        return "-1>p>-3"
    elif -300 > evaluation >= -500:
        return "-3>p>-5"
    else:
        return "<-5"


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


def create_statistics(fen_file_evaluated):
    df = pd.read_csv(fen_file_evaluated)

    # filter the mates
    df = df[~df['Evaluation'].str.startswith('#')]
    df["Evaluation"] = df["Evaluation"].astype(int)
    df["Evaluation_Predicted"] = df["Evaluation_Predicted"].astype(int)

    print(f"Creating statistics for {fen_file_evaluated}, wich are {len(df)} positions ...")

    # add class labels
    df["Evaluated_Class"] = df["Evaluation"].apply(evaluation_to_class)
    df["Evaluated_Class_Predicted"] = df["Evaluation_Predicted"].apply(evaluation_to_class)

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(7, 2, figsize=(42, 24))

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
    class_labels = ["MATE", ">5", "5>p>3", "3>p>1", "1>p>-1", "-1>p>-3", "-3>p>-5", "<-5"]
    curr_x = 3
    curr_y = 0
    for class_label in class_labels:
        plot_axes = axes[curr_x, curr_y]
        curr_x = curr_x if curr_y == 0 else curr_x + 1
        curr_y = 1 if curr_y == 0 else 0

        curr_plot = sns.countplot(x='Evaluated_Class_Predicted', data=df[df["Evaluated_Class"] == class_label], ax=plot_axes)
        plot_axes.set_title(f'Distribution of Predicted Classes for True Class {class_label}')
        plot_axes.set_xlabel('Class Label')
        plot_axes.set_ylabel('Frequency')
        write_values_in_bars(curr_plot)

    plt.tight_layout()
    # plt.show()

    plt.savefig('plot.png', bbox_inches='tight')
    # fig.savefig("out.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation helper for Chessmait")

    parser.add_argument("--fen-evaluation", action='store_true', required=False)
    parser.add_argument("--fen-evaluation-file", type=str, required=False)
    parser.add_argument("--statistics", type=str, required=False)
    parser.add_argument("--add-classes", type=str, required=False)
    args = parser.parse_args()

    model = ChessmaitMlp1()
    model.to("cuda")
    model.load_state_dict(torch.load(f"models/{MODEL_NAME}.pth"))
    model.eval()

    if args.fen_evaluation:
        evaluate_fen_command_line()
    elif args.statistics:
        create_statistics(args.statistics)
    elif args.add_classes:
        add_classes(args.add_classes)
    elif args.fen_evaluation_file:
        evaluate_fen_file(args.fen_evaluation_file)
