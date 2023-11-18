import argparse
import csv
import os.path

import chess.engine
import chess.pgn

pgn_data_path_raw = os.path.join("data", "raw", "pgn")
pgn_data_path_preprocessed = os.path.join("data", "preprocessed")
OUTPUT_FILE_NAME = "ficsgamesdb_202301_standard2000_nomovetimes_309749.csv"
NUMBER_OF_GAMES_TO_PREPROCESS = 10000  # set to -1 to preprocess all games
pgn_data_file = "ficsgamesdb_202301_standard2000_nomovetimes_309749.pgn"


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


def preprocess_pgn(path_to_chess_engine: str):
    number_of_games_preprocessed = 1

    with open(os.path.join(pgn_data_path_raw, pgn_data_file)) as pgn_file:
        while True:
            output_data = []
            game = chess.pgn.read_game(pgn_file)

            if game is None:
                break

            if number_of_games_preprocessed > 0 and number_of_games_preprocessed == NUMBER_OF_GAMES_TO_PREPROCESS:
                break

            if number_of_games_preprocessed % 100 == 0:
                print(f"Preprocessing game {number_of_games_preprocessed}")
            board = game.board()

            with chess.engine.SimpleEngine.popen_uci(path_to_chess_engine) as engine:
                for move in game.mainline_moves():
                    board.push(move)
                    fen = board.fen()

                    # Evaluate the position
                    info = engine.analyse(board, chess.engine.Limit(depth=10))

                    # Get the evaluation score
                    evaluation = info["score"].white().score()
                    # make sure evaluation is always an integer - otherwise training
                    # our regression model is not going to work
                    # usually, if there is a forced mate detected by the engine, the evaluation
                    # is not an integer any more
                    # TODO: how to handle situations like "mate in 5"?
                    if not evaluation:
                        continue

                    try:
                        int(evaluation)
                    except ValueError:
                        continue

                    output_data.append([fen, evaluation])

            number_of_games_preprocessed += 1

            # after each game, we append the content to the .csv file
            append_preprocessed_csv(os.path.join(pgn_data_path_preprocessed, OUTPUT_FILE_NAME), output_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-to-chess-engine', help='Path to your local chess engine')
    args = parser.parse_args()

    print(f"Starting to preprocess the PGN data from file {pgn_data_file} ...")
    preprocess_pgn(args.path_to_chess_engine)
