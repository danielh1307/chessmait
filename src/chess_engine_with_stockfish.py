import chess.engine
import os
import src.board_status as bs
import src.chess_engine as ce
from timeit import default_timer as timer

PATH_TO_CHESS_ENGINE = os.path.join("stockfish", "stockfish-windows-x86-64-avx2.exe")


def run():
    board = chess.Board()
    board_status = bs.BoardStatus()

    max_moves = 100

    model_white = ce.get_model_white()

    start = timer()

    wins = {True: 0, False: 0}
    with chess.engine.SimpleEngine.popen_uci(PATH_TO_CHESS_ENGINE) as engine:
        engine.configure({"UCI_Elo": 1320})

        for i in range(10000):

            board.reset_board()
            stop = False
            moves = 0
            #do_not_stop = False
            #print(f"Round: {i}")
            while not stop:
                moves += 1
                is_white = (moves % 2) == 1
                board_status.cache(board)
                #print(f"--------------- Moves: {moves:02d} {'white' if is_white else 'black'}")
                if is_white:
                    ce.play(board, model_white, is_white)
                else:
                    result = engine.play(board, chess.engine.Limit(time=0.001, depth=1, nodes=1))
                    board.push(result.move)
                #board_status.print(board)
                ''''
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
                if board.is_game_over() or moves > max_moves:
                    #print(f"--------------- Moves: {moves:02d} {'white' if is_white else 'black'} ------------")
                    stop = True

            #board_status.print(board)
            #print("--------------------------------------------")
            #print(board_status.reason_why_the_game_is_over(board, moves > max_moves))
            #print("--------------------------------------------")
            wins[is_white] += 1
            print(f"wins: white = {wins[True]} - black = {wins[False]}")
            print("", end='\r')

    end = timer()
    print(f"time: {(end - start):0.1f} sec")

if __name__ == "__main__":
    run()