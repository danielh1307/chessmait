import argparse
import os

import chess
import chess.engine
import chess.svg
import pandas as pd
import torch

from src.lib.analytics_utilities import count_pieces
from src.lib.utilities import fen_to_tensor_one_board
from src.lib.utilities import get_device
from src.model.ChessmaitMlp5 import ChessmaitMlp5

# This script lets the user prompt a FEN position. For all possible moves,
# the evaluations are predicted by a model and calculated by a (local) Stockfish
# engine. The moves are then ranked from best to worse and printed both for
# the model and Stockfish evaluation.

device = get_device()

# adjust those values based on the used model
MODEL_NAME = "lemon-plasma-103"
model = ChessmaitMlp5()


def fen_to_tensor(_fen):
    return fen_to_tensor_one_board(_fen)


def evaluate_board_by_model(_board):
    # evaluate the current position
    current_fen = _board.fen()
    current_fen_in_tensor = fen_to_tensor(current_fen).to(device)

    with torch.no_grad():
        input_batch = current_fen_in_tensor.unsqueeze(0)
        output = model(input_batch)

        return output


def evaluate_board_by_stockfish(_board, path_to_chess_engine):
    with chess.engine.SimpleEngine.popen_uci(path_to_chess_engine) as engine:
        # Evaluate the position
        info = engine.analyse(_board, chess.engine.Limit(depth=10))

        return info["score"].white().score()


def get_all_moves_evaluated(_board, path_to_chess_engine=None):
    all_possible_moves = {}

    # now we evaluate each move
    for possible_next_move in list(_board.legal_moves):
        board_copy = _board.copy()
        board_copy_san = _board.copy()
        board_copy.push_uci(possible_next_move.uci())
        next_move_evaluation_model = evaluate_board_by_model(board_copy).item()
        next_move_evaluation_stockfish = evaluate_board_by_stockfish(board_copy,
                                                                     path_to_chess_engine) if path_to_chess_engine else 0
        # make sure notation of move is algebraic
        move = chess.Move.from_uci(possible_next_move.uci())
        all_possible_moves[board_copy_san.san(move)] = {
            "model": next_move_evaluation_model,
            "stockfish": next_move_evaluation_stockfish
        }

    return all_possible_moves


def sort_evaluated_moves(_board, _all_moves, sort_by):
    all_moves_sorted = dict(sorted(_all_moves.items(), key=lambda item: item[1][sort_by], reverse=_board.turn))
    return [(key, value[sort_by]) for key, value in all_moves_sorted.items()]


def get_best_move_model(_board, winning_move):
    _all_moves = get_all_moves_evaluated(_board)
    sorted_evaluated_moves = sort_evaluated_moves(_board, _all_moves, 'model')
    evaluated_move = sorted_evaluated_moves[0][0]
    _i = 1
    for move in sorted_evaluated_moves:
        if move[0] == winning_move:
            break
        _i += 1
    return f"{evaluated_move} ({_i})"


def kaufman():
    df = pd.read_csv(os.path.join("src", "analyzing", "kaufman_test.csv"))
    df["Num_Pieces"] = df["FEN"].apply(lambda x: count_pieces(x))
    df['Evaluated_Move'] = df.apply(lambda row: get_best_move_model(chess.Board(row['FEN']), row['Winning_Move']),
                                    axis=1)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df[['id', 'Winning_Move', 'Evaluated_Move', 'Num_Pieces']])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-to-chess-engine', help='Path to your local chess engine', required=True)
    args = parser.parse_args()

    model.to(device)
    model.load_state_dict(torch.load(f"models/{MODEL_NAME}.pth", map_location=device))
    model.eval()

    while True:
        fen = input("What is your FEN position ('kaufman' for kaufman test or 'q' to quit)?")
        if fen == "q":
            break
        elif fen == "kaufman":
            kaufman()
            continue

        board = chess.Board(fen)

        all_moves = get_all_moves_evaluated(board, args.path_to_chess_engine)
        all_moves_sorted_model_array = sort_evaluated_moves(board, all_moves, 'model')
        all_moves_sorted_stockfish_array = sort_evaluated_moves(board, all_moves, 'stockfish')

        print("Rank\tModel move\tModel eval\tStockfish move\tStockfish eval")
        for i in range(len(all_moves_sorted_model_array)):
            move_model = all_moves_sorted_model_array[i][0]
            eval_model = format(all_moves_sorted_model_array[i][1], ".2f")
            move_stockfish = all_moves_sorted_stockfish_array[i][0]
            eval_stockfish = format(all_moves_sorted_stockfish_array[i][1], ".2f")
            print(f"{i + 1}.\t\t{move_model}\t\t{eval_model}\t\t{move_stockfish}\t\t\t{eval_stockfish}")
