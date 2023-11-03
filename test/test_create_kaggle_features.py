import os
import chess.pgn

from src.preprocessing.preprocess_kaggle import get_move_scores, get_number_of_half_moves


def test_get_move_scores():
    # arrange
    csv_file = os.path.join("test", "resources", "test-stockfish.csv")

    # act
    move_scores = get_move_scores(csv_file)

    # assert
    assert len(move_scores) == 3


def test_get_number_of_moves():
    # arrange
    number_of_moves = []
    with open(os.path.join("test", "resources", "test-data.pgn")) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)

            if game is None:
                break

            # act
            number_of_moves.append(get_number_of_half_moves(game))

    # assert
    assert number_of_moves[0] == 38
    assert number_of_moves[1] == 13
    assert number_of_moves[2] == 106



