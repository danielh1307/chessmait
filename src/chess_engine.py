import sys

import torch
import os
import chess
import time
from random import randrange
from lib.position_validator import get_valid_positions
from src.model.ChessmaitMlp1 import ChessmaitMlp1
from src.model.rbf1 import RbfNetwork1
from src.lib.utilities import fen_to_tensor_one_board
from src.lib.utilities import is_checkmate

PATH_TO_MODEL = os.path.join("models")
MODEL_FILE_WHITE = "firm-star-24.pth"
MODEL_FILE_BLACK = "smart-valley-6.pth"

model_white = ChessmaitMlp1()
model_white.load_state_dict(torch.load(os.path.join(PATH_TO_MODEL, MODEL_FILE_WHITE)))
model_white.eval()

model_black = ChessmaitMlp1()
model_black.load_state_dict(torch.load(os.path.join(PATH_TO_MODEL, MODEL_FILE_BLACK)))
model_black.eval()

#normalized_evaluation = (evaluation - MIN_EVALUATION) / (MAX_EVALUATION - MIN_EVALUATION)
MAX_EVALUATION = 12352
MIN_EVALUATION = -12349

def get_valid_moves_with_evaluation(current_position, is_white):
    valid_positions = get_valid_positions(current_position)
    dict = {}
    for valid_position in valid_positions:
        t = fen_to_tensor_one_board(valid_position)
        t = t.view(-1, t.size(0))
        normalized_evaluation = model_white(t) if is_white else model_black(t)
        evaluation = normalized_evaluation * (MAX_EVALUATION - MIN_EVALUATION) + MIN_EVALUATION
        evaluation = evaluation.item()
        if evaluation in dict:
            dict[evaluation].append(valid_position)
        else:
            dict[evaluation] = [valid_position]
    return dict

def get_best_move(current_position, is_white):
    valid_moves_with_evaluation = get_valid_moves_with_evaluation(current_position, is_white)
    min_max_eval_key = next(iter(valid_moves_with_evaluation))
    for k in valid_moves_with_evaluation.keys():
        if (is_white and k > min_max_eval_key) or (not is_white and k < min_max_eval_key):
            min_max_eval_key = k
    try:
        index = 0 if len(valid_moves_with_evaluation[min_max_eval_key]) == 1 else randrange(len(valid_moves_with_evaluation[min_max_eval_key]) - 1)
        return min_max_eval_key, valid_moves_with_evaluation[min_max_eval_key][index]
    except:
        print("xxx")

# colors and font settings see: https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
COLOR_MOVES = '\033[1;91m'
EMPTY = '\033[37m'
DEFAULT = '\033[0m'

def convert_to_int(board):
    indices = '♚♛♜♝♞♟⭘♙♘♗♖♕♔'
    unicode = board.unicode()
    return [
        [indices.index(c)-6 for c in row.split()]
        for row in unicode.split('\n')
    ]

mapped = {
         1:'P',     # White Pawn
        -1:'p',    # Black Pawn
         2:'N',     # White Knight
        -2:'n',    # Black Knight
         3:'B',     # White Bishop
        -3:'b',    # Black Bishop
         4:'R',     # White Rook
        -4:'r',    # Black Rook
         5:'Q',     # White Queen
        -5:'q',    # Black Queen
         6:'K',     # White King
        -6:'k',    # Black King
         0:'.',
        }

def convert_to_str(new_board, old_board):
    for row in range(8):
        for col in range(8):
            if new_board[row][col] != old_board[row][col]:
                print(f"{COLOR_MOVES}{mapped[new_board[row][col]]}{DEFAULT} ", end='')
            elif mapped[new_board[row][col]] == '.':
                print(f"{EMPTY}{mapped[new_board[row][col]]}{DEFAULT} ", end='')
            else:
                print(f"{mapped[new_board[row][col]]} ", end='')
        print("")

board = chess.Board()
print(board)

abort = False
round = 0
evals = {True:{}, False:{}}
while not abort:
    round += 1
    is_white = (round % 2) == 1
    old_board = convert_to_int(board)
    print(f"--------------- Round: {round:02d} {'white' if is_white else 'black'} --- is checkmate: {board.is_checkmate()}", end='')
    fen = board.fen()
    best_move_eval, best_move_fen = get_best_move(fen, is_white)
    print(f" --- eval: {best_move_eval}")
    board.set_fen(best_move_fen)
    new_board = convert_to_int(board)
    convert_to_str(new_board, old_board)
    #print('continue (y/n)? ', end='')
    #i = str(input())
    #if i == 'n':
    #    abort = True
    #time.sleep(1)
    if best_move_eval in evals[is_white]:
        evals[is_white][best_move_eval] += 1
    else:
        evals[is_white][best_move_eval] = 1
    if board.is_checkmate() or board.is_stalemate() or round > 1000:
        abort = True

print(f"evaluations white: {evals[True]}")
print(f"evaluations black: {evals[False]}")
