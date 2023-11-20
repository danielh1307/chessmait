import csv
import os.path
import re

import chess.engine
import chess.pgn

# TODO add path
pgn_data_path_raw = ""
pgn_data_path_preprocessed = os.path.join("data", "preprocessed")
NUMBER_OF_GAMES_TO_PREPROCESS = -1  # set to -1 to preprocess all games
EVAL_REGEX_NORMAL_MOVE = r'\[%eval ([-+]?\d+\.\d+)]'
EVAL_REGEX_MATE = r'\[%eval (#[-+]?\d+)]'
HAS_EVAL = '%eval'


def append_preprocessed_csv(csv_file: str, data: list):
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
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)  # Write the data rows


def has_evaluation(game):
    for node in game.mainline():
        if node.comment and HAS_EVAL in node.comment:
            return True
    return False


def preprocess_lichess():
    number_of_games_preprocessed = 1

    with open(os.path.join(pgn_data_path_raw, pgn_data_file)) as pgn_file:
        while True:
            output_data = []
            game = chess.pgn.read_game(pgn_file)

            if game is None:
                break

            if number_of_games_preprocessed > 0 and number_of_games_preprocessed == NUMBER_OF_GAMES_TO_PREPROCESS:
                break

            if not has_evaluation(game):
                continue

            if number_of_games_preprocessed % 100 == 0:
                print(f"Preprocessing game {number_of_games_preprocessed}")

            board = game.board()
            for node in game.mainline():
                board.push(node.move)
                eval_match = re.search(EVAL_REGEX_NORMAL_MOVE, str(node.comment))
                if eval_match:
                    evaluation = eval_match.group(1)
                    # Lichess has evaluations based on pawns, but we use cp instead
                    evaluation = int(float(evaluation) * 100)
                else:
                    eval_match = re.search(EVAL_REGEX_MATE, str(node.comment))
                    if eval_match:
                        evaluation = eval_match.group(1)
                    else:
                        if board.is_checkmate():
                            evaluation = "##"
                        else:
                            print("Unknown evaluation: ", str(node.comment))
                            print("game: ", game.headers["Site"])
                fen = board.fen()

                output_data.append([fen, evaluation])
            number_of_games_preprocessed += 1

            # after each game, we append the content to the .csv file
            append_preprocessed_csv(os.path.join(pgn_data_path_preprocessed, pgn_data_file[0:-4] + ".csv"), output_data)


if __name__ == "__main__":
    pgn_files = [file for file in os.listdir(pgn_data_path_raw) if file.endswith('.pgn')]
    for pgn_data_file in pgn_files:
        print(f"Starting to preprocess the PGN data from file {pgn_data_file} ...")
        preprocess_lichess()
    print("Finished")
