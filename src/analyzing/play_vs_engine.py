import random

import chess
import torch

from src.lib.utilities import fen_to_bitboard
from src.lib.utilities import get_device, is_checkmate
from src.model.ChessmaitCnn4Bitboard import ChessmaitCnn4Bitboard

# This script allows you to play vs. one of our models.
# Just fill the variables accordingly to your needs.

model = ChessmaitCnn4Bitboard()
model.load_state_dict(torch.load("models/fresh-blaze-174.pth"))
model.eval()

NORMALIZATION_USED = False
MAX_EVALUATION = 12352
MIN_EVALUATION = -12349
CHESSMAIT_PLAYS_WHITE = True


def reverse_normalisiation(normalized_evaluation):
    if NORMALIZATION_USED:
        return normalized_evaluation * (MAX_EVALUATION - MIN_EVALUATION) + MIN_EVALUATION
    return normalized_evaluation


def evaluate_board(board, device):
    # evaluate the current posision
    current_fen = board.fen()
    current_fen_in_tensor = fen_to_bitboard(current_fen).to(device)

    with torch.no_grad():
        input_batch = current_fen_in_tensor.unsqueeze(0)
        output = model(input_batch)

        return output


def get_next_move_model(board, device):
    all_possible_moves = []

    # now we evaluate each move
    for possible_next_move in list(board.legal_moves):
        board_copy = board.copy()
        board_copy.push_uci(possible_next_move.uci())
        if is_checkmate(board_copy.fen()):
            # a checkmate is always the best move
            return possible_next_move

        next_move_evaluation = reverse_normalisiation(evaluate_board(board_copy, device).item())
        all_possible_moves.append((possible_next_move, next_move_evaluation))

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
