import os
import src.board_status as bs
from stockfish import Stockfish
import src.chess_engine as ce
import chess
from timeit import default_timer as timer

PATH_TO_CHESS_ENGINE = os.path.join("stockfish", "stockfish-windows-x86-64-avx2.exe")

# skill level | time | depth | skill level int | elo
# one           50     5       -9                800
# two           100    5       -5                1100
# three         150    5       -1                1400
# four          200    5       +3                1700
# five          300    5       +7                2000
# six           400    8       +11               2300
# seven         500    13      +16               2700
# eight         1000   22      +20               3000

def run():
    board = chess.Board()
    board_status = bs.BoardStatus()

    do_not_stop = False
    max_rounds = 100

    start = timer()

    model_white = ce.get_model_white()

    stockfish = Stockfish(path=PATH_TO_CHESS_ENGINE)
    #stockfish.set_elo_rating(1000)
    stockfish.set_skill_level(0)
    #stockfish.set_depth()

    wins = {True: 0, False: 0}
    for i in range(100):

        board.reset_board()
        stop = False
        round = 0
        do_not_stop = False

        while not stop:
            round += 1
            is_white = (round % 2) == 1
            board_status.cache(board)
            #print(f"--------------- Round: {round:02d} {'white' if is_white else 'black'}")
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
            #board_status.print(board)
            '''
            if not do_not_stop:
                console_input = input("continue/stop/automatic/max-rounds ('Enter'/s/a/#)? ")
                if console_input == 's':
                    stop = True
                elif console_input == 'a':
                    do_not_stop = True
                elif console_input != "" and console_input.isdigit():
                    max_rounds = int(console_input)
                    do_not_stop = True
            '''
            if board.is_game_over() or round > max_rounds:
                #print(f"--------------- Round: {round:02d} {'white' if is_white else 'black'}")
                stop = True

        #board_status.print(board)
        #print("--------------------------------------------")
        #print(board_status.reason_why_the_game_is_over(board, round > max_rounds))
        #print("--------------------------------------------")
        wins[is_white] += 1
        print(f"wins: white = {wins[True]} - black = {wins[False]}")

    end = timer()
    print(f"time: {(end - start):0.1f} sec")

if __name__ == "__main__":
    run()