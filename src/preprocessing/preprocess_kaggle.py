import csv
import os

import chess.pgn

kaggle_data_path_raw = os.path.join("..", "..", "data", "raw", "kaggle")
kaggle_data_path_preprocessed = os.path.join("..", "..", "data", "preprocessed", "kaggle")
OUTPUT_FILE_NAME = "kaggle_preprocessed.csv"
NUMBER_OF_GAMES_TO_PREPROCESS = -1  # set to -1 to preprocess all games


def get_move_scores(csv_file: str) -> dict:
    """
    This method reads the .csv file with the move scores as provided by Kaggle and returns a
    dictionary with the content.

    Parameters
    ----------
    csv_file : str
        The .csv file with the move scores.

    Returns
    -------
    dict
        Key: the id of the game (type string)
        Value: the list of the scores (list of strings)

    """
    result_dict = {}

    with open(csv_file, newline="") as score_file:
        reader = csv.DictReader(score_file)
        for row in reader:
            event = str(row["Event"])
            move_scores = [str(score) for score in row["MoveScores"].split()]
            result_dict[event] = move_scores

    return result_dict


def get_number_of_half_moves(game) -> int:
    """
    Count the number of half-moves in the given game. A half-move is one move by either white or black.

    Parameters
    ----------
    game

    Returns
    -------
    int
        Number of moves in the given game.

    """
    num_of_moves = 0
    for _ in game.mainline_moves():
        num_of_moves += 1
    return num_of_moves


def write_preprocessed_csv(csv_file: str, data: list):
    """
    Write the preprocessed .csv file.

    Parameters
    ----------
    csv_file : str
        The path to the .csv file which is written.
    data : list
        The array with data which is written.


    Returns
    -------

    """
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['FEN', 'Evaluation'])  # Write header row
        writer.writerows(data)  # Write the data rows


def preprocess_kaggle():
    """
    This method preprocesses the Kaggle data. The files are taken from https://www.kaggle.com/competitions/finding-elo
    and remain unchanged.

    The data consists of 50.000 chess games in pgn format and the stockfish evaluation of every move.
    Preprocessing means we want to have a format which we can take as input to train a neural network.
    So for every half-move, we make the FEN and take the corresponding evaluation and store them to a .csv file.

    Returns
    -------

    """
    result_dict = get_move_scores(os.path.join(kaggle_data_path_raw, "stockfish.csv"))
    output_data = []

    number_of_games_preprocessed = 0
    number_of_evaluations = 0
    with open(os.path.join(kaggle_data_path_raw, "data.pgn")) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)

            if game is None:
                break

            if number_of_games_preprocessed > 0 and number_of_games_preprocessed == NUMBER_OF_GAMES_TO_PREPROCESS:
                break

            event_number = game.headers["Event"]

            if int(event_number) % 1000 == 0:
                print(f"Preprocessing Event {event_number} ...")
                print("-" * 40)

            board = game.board()
            number_of_half_moves = get_number_of_half_moves(game)

            if number_of_half_moves < 3:
                # if we have less than three moves, we skip this event
                print(f"We skip event {event_number} because it contains only {number_of_half_moves} half-move(s) ...")
                continue

            # get the result to this game
            if not result_dict[event_number]:
                raise Exception(f"No result found for Event {event_number}")

            number_of_scores = len(result_dict[event_number])

            # check if we have one score per move
            if number_of_half_moves != number_of_scores:
                raise Exception(f"Event {event_number} has {number_of_half_moves} moves but {number_of_scores} scores")

            move_number = 0
            for move in game.mainline_moves():
                board.push(move)
                fen = board.fen()
                evaluation = result_dict[event_number][move_number]
                # make sure evaluation is always an integer - otherwise training
                # our regression model is not going to work
                # usually, if there is a forced mate detected by the engine, the evaluation
                # is not an integer any more
                # TODO: how to handle situations like "mate in 5"?
                try:
                    int(evaluation)
                except ValueError:
                    continue

                output_data.append([fen, evaluation])
                move_number += 1
                number_of_evaluations += 1
            number_of_games_preprocessed += 1

        write_preprocessed_csv(os.path.join(kaggle_data_path_preprocessed, OUTPUT_FILE_NAME), output_data)

    print(f"Number of games preprocessed: {number_of_games_preprocessed}, "
          f"number of evaluations: {number_of_evaluations}")


if __name__ == "__main__":
    print("Starting to preprocess the Kaggle data ...")
    if NUMBER_OF_GAMES_TO_PREPROCESS > 0:
        print(f"Preprocessing {NUMBER_OF_GAMES_TO_PREPROCESS} games ...")
    preprocess_kaggle()
