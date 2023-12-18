import os
import play.board_status as bs
from stockfish import Stockfish
import play.chess_engine as ce
import play.trained_model as tm
import chess
from timeit import default_timer as timer

PATH_TO_CHESS_ENGINE = os.path.join("stockfish", "stockfish-windows-x86-64-avx2.exe")

'''
skill level | time | depth | skill level int | elo
one           50     5       -9                800
two           100    5       -5                1100
three         150    5       -1                1400
four          200    5       +3                1700
five          300    5       +7                2000
six           400    8       +11               2300
seven         500    13      +16               2700
eight         1000   22      +20               3000
'''

def run():
    board = chess.Board()
    board_status = bs.BoardStatus()

    model_white = tm.model_wild_snow_28

    start = timer()

    wins = {True: 0, False: 0}
    stockfish = Stockfish(path=PATH_TO_CHESS_ENGINE)
    stockfish.set_skill_level(0)

    automatic = False
    stop = False

    for i in range(1, 11):

        board.reset_board()
        one_round = False

        moves = 0
        print(f"Round: {i}")
        while not stop:
            moves += 1
            is_white = (moves % 2) == 1
            board_status.cache(board)
            print(f"--------------- Round: {i} --- Moves: {moves:02d} {'white' if is_white else 'black'}")
            if is_white:
                ce.play(board, model_white, is_white)
            else:
                stockfish.set_fen_position(board.fen())
                best_move = stockfish.get_best_move()
                if best_move == None:
                    message = "game over: winner is " + ("black" if is_white else "white")
                    print(message)
                    break
                board.push_uci(best_move)
            board_status.print(board)
            if not one_round and not automatic:
                print("one-move:  'Enter'")
                print("stop:      s")
                print("one-round: r")
                print("automatic: a")
                console_input = input("how to proceed ?")
                if console_input == 's':
                    stop = True
                elif console_input == 'r':
                    one_round = True
                elif console_input == 'a':
                    automatic = True
            if board.is_game_over():
                reason = board_status.reason_why_the_game_is_over(board)
                print("--------------------------------------------")
                print(reason)
                if reason == "Termination.CHECKMATE":
                    wins[is_white] += 1
                break

        if not stop:
            print(f"wins: white = {wins[True]} - black = {wins[False]}")
            print("--------------------------------------------")
        else:
            break

    if stop:
        print("stopped")

    end = timer()
    print(f"time: {(end - start):0.1f} sec")


if __name__ == "__main__":
    run()