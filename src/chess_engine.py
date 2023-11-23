import random

import chess
from lib.position_validator import get_valid_positions
from src.lib.utilities import fen_to_tensor_one_board
from src.lib.utilities import fen_to_cnn_tensor_non_hot_enc
import src.board_status as bs
import trained_model as tm
import torch as tc


def get_valid_moves_with_evaluation(current_position, trained_model):
    valid_positions = get_valid_positions(current_position)
    dict = {}
    for valid_position in valid_positions:
        if trained_model.fen_to_tensor == 1:
            t = fen_to_cnn_tensor_non_hot_enc(valid_position)
            t = tc.unsqueeze(t, dim=0)
        else:
            t = fen_to_tensor_one_board(valid_position)
            t = t.view(-1, t.size(0))
        normalized_evaluation = trained_model.model(t)
        evaluation = trained_model.de_normalize(normalized_evaluation)
        evaluation = evaluation.item()
        if evaluation in dict:
            dict[evaluation].append(valid_position)
        else:
            dict[evaluation] = [valid_position]
    return dict


def get_best_move(current_position, model, is_white):
    valid_moves_with_evaluation = get_valid_moves_with_evaluation(current_position, model)
    min_max_eval_key = next(iter(valid_moves_with_evaluation))
    for k in valid_moves_with_evaluation.keys():
        if (is_white and k > min_max_eval_key) or (not is_white and k < min_max_eval_key):
            min_max_eval_key = k
    index = 0 if len(valid_moves_with_evaluation[min_max_eval_key]) == 1 else random.randrange(
        len(valid_moves_with_evaluation[min_max_eval_key]) - 1)
    return min_max_eval_key, valid_moves_with_evaluation[min_max_eval_key][index]


def play(board, model, is_white):
    fen = board.fen()
    best_move_eval, best_move_fen = get_best_move(fen, model, is_white)
    board.set_fen(best_move_fen)
    return best_move_eval


def run():
    board = chess.Board()
    board_status = bs.BoardStatus()

    stop = False
    one_round = False
    moves = 0
    evals = {True: {}, False: {}}

    model_white = tm.model_fluent_mountain_47
    model_black = tm.model_wild_snow_28

    while not stop:
        moves += 1
        is_white = (moves % 2) == 1
        board_status.cache(board)
        print(f"--------------- Moves: {moves:02d} {'white' if is_white else 'black'}", end='')
        best_move_eval = play(board, model_white if is_white else model_black, is_white)
        print(f" --- eval: {best_move_eval}")
        board_status.print(board)
        if not one_round:
            print("one-move:  'Enter'")
            print("stop:      s")
            print("one-round: r")
            console_input = input("how to proceed ?")
            if console_input == 's':
                stop = True
            elif console_input == 'r':
                one_round = True
        if best_move_eval in evals[is_white]:
            evals[is_white][best_move_eval] += 1
        else:
            evals[is_white][best_move_eval] = 1
        if board.is_game_over():
            break

    if not stop:
        print("--------------------------------------------")
        print(board_status.reason_why_the_game_is_over(board))
        print("--------------------------------------------")
        print(f"evaluations white: {evals[True]}")
        print(f"evaluations black: {evals[False]}")
    else:
        print("stopped")


if __name__ == "__main__":
    run()
