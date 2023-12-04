import argparse

import chess
import chess.engine
import chess.svg
import torch

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


def get_next_move_model(_board, path_to_chess_engine):
    all_possible_moves = {}

    # now we evaluate each move
    for possible_next_move in list(_board.legal_moves):
        board_copy = _board.copy()
        board_copy.push_uci(possible_next_move.uci())
        next_move_evaluation_model = evaluate_board_by_model(board_copy).item()
        next_move_evaluation_stockfish = evaluate_board_by_stockfish(board_copy, path_to_chess_engine)
        all_possible_moves[possible_next_move.uci()] = {
            "model": next_move_evaluation_model,
            "stockfish": next_move_evaluation_stockfish
        }

    return all_possible_moves


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-to-chess-engine', help='Path to your local chess engine', required=True)
    args = parser.parse_args()

    model.to(device)
    model.load_state_dict(torch.load(f"models/{MODEL_NAME}.pth", map_location=device))
    model.eval()

    while True:
        fen = input("What is your FEN position (q to quit)?")
        if fen == "q":
            break
        board = chess.Board(fen)

        all_moves = get_next_move_model(board, args.path_to_chess_engine)
        all_moves_sorted_model = dict(sorted(all_moves.items(), key=lambda item: item[1]['model'], reverse=board.turn))
        all_moves_sorted_stockfish = dict(
            sorted(all_moves.items(), key=lambda item: item[1]['stockfish'], reverse=board.turn))

        all_moves_sorted_model_array = [(key, value['model']) for key, value in all_moves_sorted_model.items()]
        all_moves_sorted_stockfish_array = [(key, value['stockfish']) for key, value in
                                            all_moves_sorted_stockfish.items()]

        print("Rank\tModel move\tModel eval\tStockfish move\tStockfish eval")
        for i in range(len(all_moves_sorted_model_array)):
            move_model = all_moves_sorted_model_array[i][0]
            eval_model = format(all_moves_sorted_model_array[i][1], ".2f")
            move_stockfish = all_moves_sorted_stockfish_array[i][0]
            eval_stockfish = format(all_moves_sorted_stockfish_array[i][1], ".2f")
            print(f"{i + 1}.\t\t{move_model}\t\t{eval_model}\t\t{move_stockfish}\t\t\t{eval_stockfish}")