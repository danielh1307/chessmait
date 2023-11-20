import torch
import os
import chess
from random import randrange
from lib.position_validator import get_valid_positions
from src.model.ChessmaitMlp1 import ChessmaitMlp1
from src.lib.utilities import fen_to_tensor_one_board
import src.board_status as bs

PATH_TO_MODEL = os.path.join("models")
MODEL_FILE_WHITE = "firm-star-24.pth"
MODEL_FILE_BLACK = "smart-valley-6.pth"

model_white = ChessmaitMlp1()
model_white.load_state_dict(torch.load(os.path.join(PATH_TO_MODEL, MODEL_FILE_WHITE)))
model_white.eval()

model_black = ChessmaitMlp1()
model_black.load_state_dict(torch.load(os.path.join(PATH_TO_MODEL, MODEL_FILE_BLACK)))
model_black.eval()

# normalized_evaluation = (evaluation - MIN_EVALUATION) / (MAX_EVALUATION - MIN_EVALUATION)
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
    index = 0 if len(valid_moves_with_evaluation[min_max_eval_key]) == 1 else randrange(
        len(valid_moves_with_evaluation[min_max_eval_key]) - 1)
    return min_max_eval_key, valid_moves_with_evaluation[min_max_eval_key][index]

def run():
    board = chess.Board()
    board_status = bs.BoardStatus()

    stop = False
    do_not_stop = False
    round = 0
    max_rounds = 10
    evals = {True: {}, False: {}}

    while not stop:
        round += 1
        is_white = (round % 2) == 1
        board_status.cache(board)
        print(f"--------------- Round: {round:02d} {'white' if is_white else 'black'}", end='')
        fen = board.fen()
        best_move_eval, best_move_fen = get_best_move(fen, is_white)
        print(f" --- eval: {best_move_eval}")
        board.set_fen(best_move_fen)
        board_status.print(board)
        if not do_not_stop:
            console_input = input("continue/stop/automatic/max-rounds ('Enter'/s/a/#)? ")
            if console_input == 's':
                stop = True
            elif console_input == 'a':
                do_not_stop = True
            elif console_input != "" and console_input.isdigit():
                max_rounds = int(console_input)
                do_not_stop = True
        if best_move_eval in evals[is_white]:
            evals[is_white][best_move_eval] += 1
        else:
            evals[is_white][best_move_eval] = 1
        if board.is_game_over() or round > max_rounds:
            stop = True

    print("--------------------------------------------")
    print(board_status.reason_why_the_game_is_over(board, round > max_rounds))
    print("--------------------------------------------")
    print(f"evaluations white: {evals[True]}")
    print(f"evaluations black: {evals[False]}")

if __name__ == "__main__":
    run()
