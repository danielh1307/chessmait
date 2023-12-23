import sys

import chess

import play.board_status as bs
import play.trained_model as tm
from src.lib.fen_to_tensor import fen_to_tensor_one_board, fen_to_cnn_tensor_non_hot_enc, fen_to_bitboard
from src.lib.utilities import is_stalemate, is_checkmate, get_valid_positions


def get_valid_moves_with_evaluation(current_position, trained_model):
    valid_positions = get_valid_positions(current_position)
    dict = {}
    for valid_position in valid_positions:
        if trained_model.fen_to_tensor == 'fen_to_cnn_tensor_non_hot_enc':
            t = fen_to_cnn_tensor_non_hot_enc(valid_position)
        elif trained_model.fen_to_tensor == 'fen_to_bitboard':
            t = fen_to_bitboard(valid_position)
        elif trained_model.fen_to_tensor == 'fen_to_tensor_one_board':
            t = fen_to_tensor_one_board(valid_position)
        else:
            raise Exception(f'Unknown fen_to_tensor method: {trained_model.fen_to_tensor}')
        normalized_evaluation = trained_model.model(t.unsqueeze(0))
        evaluation = trained_model.de_normalize(normalized_evaluation)
        evaluation = evaluation.item()
        if evaluation in dict:
            dict[evaluation].append(valid_position)
        else:
            dict[evaluation] = [valid_position]
    return dict


def get_best_move(current_position, model, is_white, last_fen=None):
    valid_positions = get_valid_positions(current_position)
    valid_position_to_evalution = []

    # check if we have direct move repetitions
    repeated_move = None
    if last_fen is not None and len(last_fen) >= 5:
        if current_position.split(maxsplit=1)[0] == last_fen[-5].split(maxsplit=1)[0]:
            repeated_move = last_fen[-4].split(maxsplit=1)[0]

    for valid_position in valid_positions:
        if is_checkmate(valid_position):
            # this is always the best decision
            evaluation = sys.maxsize if is_white else sys.maxsize * (-1)
            return evaluation, valid_position
        elif is_stalemate(valid_position):
            # this is always the worst decision
            continue
        # check the after next moves, these are the moves after valid_position
        valid_after_moves_with_evaluation = get_valid_moves_with_evaluation(valid_position, model)
        # if we play white, we want the lowest of the numbers, otherwise the
        # highest (because our opponent is supposed to make the best move)
        relevant_evaluation = min(valid_after_moves_with_evaluation.keys()) if is_white else max(
            valid_after_moves_with_evaluation.keys())

        if valid_position.split(maxsplit=1)[0] == repeated_move:
            # if there is a move repetition, we evaluate it with 0
            # that means we favor every move that is positively ranked
            # if all moves are negative for us, we do the move repetition
            valid_position_to_evalution.append((0, valid_position))
        else:
            valid_position_to_evalution.append((relevant_evaluation, valid_position))

    relevant_tuple = max(valid_position_to_evalution, key=lambda x: x[0]) if is_white else min(
        valid_position_to_evalution, key=lambda x: x[0])
    return relevant_tuple


def fen_to_move(start_fen, end_fen):
    board = chess.Board(start_fen)
    for move in board.legal_moves:
        board.push(move)
        if board.fen() == end_fen:
            return move
        board.pop()
    return None


def play_return_move(board, model, is_white, last_fen=None):
    fen = board.fen()
    best_move_eval, best_move_fen = get_best_move(fen, model, is_white, last_fen)
    board.set_fen(best_move_fen)
    move = fen_to_move(fen, best_move_fen)
    return move.uci(), best_move_fen


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
