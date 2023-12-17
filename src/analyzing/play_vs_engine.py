import random
import sys

import chess
import torch

from src.lib.utilities import fen_to_tensor_one_board
from src.lib.utilities import get_device, is_checkmate, is_stalemate
from src.model.ChessmaitMlp5 import ChessmaitMlp5

# This script allows you to play vs. one of our models.
# Just fill the variables accordingly to your needs.

model = ChessmaitMlp5()
model.load_state_dict(torch.load("models/apricot-armadillo-167.pth"))
model.eval()

NORMALIZATION_USED = False
MAX_EVALUATION = 12352
MIN_EVALUATION = -12349
CHESSMAIT_PLAYS_WHITE = False


def reverse_normalisiation(normalized_evaluation):
    if NORMALIZATION_USED:
        return normalized_evaluation * (MAX_EVALUATION - MIN_EVALUATION) + MIN_EVALUATION
    return normalized_evaluation


def evaluate_board(board, device):
    # evaluate the current posision
    current_fen = board.fen()
    current_fen_in_tensor = fen_to_tensor_one_board(current_fen).to(device)

    with torch.no_grad():
        input_batch = current_fen_in_tensor.unsqueeze(0)
        output = model(input_batch)

        return output


def get_next_moves_with_evaluation(_board, _device):
    all_possible_moves = []
    for possible_next_move in list(_board.legal_moves):
        board_copy = _board.copy()
        board_copy.push_uci(possible_next_move.uci())
        if is_checkmate(board_copy.fen()):
            # a checkmate is always the best move
            next_move_evaluation = sys.maxsize if _board.turn == chess.WHITE else sys.maxsize * (-1)
            all_possible_moves.append((possible_next_move, next_move_evaluation))
            return all_possible_moves
        elif is_stalemate(board_copy.fen()):
            # we do not want to stalemate the opponent
            next_move_evaluation = sys.maxsize if _board.turn == chess.BLACK else sys.maxsize * (-1)
            all_possible_moves.append((possible_next_move, next_move_evaluation))
        else:
            all_possible_after_next_moves = []
            # we calculate the evaluation of the next move
            for possible_after_next_move in list(board_copy.legal_moves):
                board_second_copy = board_copy.copy()
                board_second_copy.push_uci(possible_after_next_move.uci())
                # TODO add checkmate
                after_next_move_evaluation = reverse_normalisiation(evaluate_board(board_second_copy, _device).item())
                #print(f"{possible_next_move} Possible after next move: {possible_after_next_move}, evaluation: {after_next_move_evaluation}")
                all_possible_after_next_moves.append((possible_next_move, after_next_move_evaluation))

            # our next_move_evaluation is the best score from all all_possible_after_next_moves
            if CHESSMAIT_PLAYS_WHITE:
                #print(all_possible_after_next_moves)
                next_move_evaluation = min(all_possible_after_next_moves, key=lambda x: x[1])[1]
            else:
                next_move_evaluation = max(all_possible_after_next_moves, key=lambda x: x[1])[1]
            # next_move_evaluation = reverse_normalisiation(evaluate_board(board_copy, _device).item())
        #print(f"Possible next move: {possible_next_move}, evaluation: {next_move_evaluation}")
        all_possible_moves.append((possible_next_move, next_move_evaluation))
    return all_possible_moves


def get_next_move_model(_board, _device):
    all_possible_moves = get_next_moves_with_evaluation(_board, _device)

    # get the best evaluation
    if CHESSMAIT_PLAYS_WHITE:
        best_evaluation = max(all_possible_moves, key=lambda x: x[1])[1]
    else:
        best_evaluation = min(all_possible_moves, key=lambda x: x[1])[1]

    # create a list of all tuples with that evaluation
    best_moves = [m for m in all_possible_moves if m[1] == best_evaluation]

    print(f"Best evaluation is {best_evaluation} and I have a list of {len(best_moves)} moves")
    if len(best_moves) == 1:
        return best_moves[0][0]
    else:
        return random.choice(best_moves)[0]


def get_next_move_random(board):
    legal_moves = list(board.legal_moves)
    random_move = random.choice(legal_moves)
    return random_move


if __name__ == "__main__":
    board = chess.Board()

    device = get_device()
    model.to(device)

    while True:
        print(board)

        if (board.turn == chess.WHITE and not CHESSMAIT_PLAYS_WHITE) or (
                board.turn != chess.WHITE and CHESSMAIT_PLAYS_WHITE):
            next_move = input("What is your next move?")
            if next_move == "q":
                break
        else:
            next_move = str(get_next_move_model(board, device))
            print(f"My move is {next_move}")

        try:
            board.push_san(next_move)
        except chess.InvalidMoveError:
            print("Invalid move. Try again.")
        except chess.IllegalMoveError:
            print("Invalid move. Try again.")
