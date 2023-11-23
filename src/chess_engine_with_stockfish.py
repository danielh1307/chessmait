import chess.engine
import os
import src.board_status as bs
import src.chess_engine as ce
import src.trained_model as tm
from timeit import default_timer as timer

PATH_TO_CHESS_ENGINE = os.path.join("stockfish", "stockfish-windows-x86-64-avx2.exe")


def run():
    board_status = bs.BoardStatus()

    # model_white = tm.model_smart_valley_6
    model_black = tm.model_upbeat_cloud_79

    start = timer()
    no_display = False

    wins = {True: 0, False: 0, 'Draw': 0}
    with chess.engine.SimpleEngine.popen_uci(PATH_TO_CHESS_ENGINE) as engine:
        engine.configure({"UCI_Elo": 1320})

        automatic = False
        stop = False

        for i in range(1, 101):

            board = chess.Board()
            one_round = False
            stop = False

            moves = 0
            if not no_display:
                print(f"Round: {i}")
            while not stop:
                moves += 1
                is_white = (moves % 2) == 1
                board_status.cache(board)
                if not no_display:
                    print(f"--------------- Round: {i} --- Moves: {moves:02d} {'white' if is_white else 'black'}", end='')
                if is_white:
                    # Stockfish
                    (eval, _) = ce.get_best_move(board.fen(), model_black, False)

                    eval_engine = engine.analyse(board, chess.engine.Limit(depth=10))["score"].white().score()

                    # Own model
                    # eval = ce.play(board, model_white, is_white)

                    if not no_display:
                        print(f" --- eval model/engine w/b: {eval}/{eval_engine}")

                    # Stockfish
                    result = engine.play(board, chess.engine.Limit(time=0.001, depth=1, nodes=1))
                    board.push(result.move)
                else:
                    # Stockfish
                    # (eval, _) = ce.get_best_move(board.fen(), model_white, False)

                    eval_engine = engine.analyse(board, chess.engine.Limit(depth=10))["score"].black().score()

                    # Own model
                    eval = ce.play(board, model_black, is_white)

                    if not no_display:
                        print(f" --- eval model/engine: {eval}/{eval_engine}")

                    # Stockfish
                    # result = engine.play(board, chess.engine.Limit(time=0.001, depth=1, nodes=1))
                    # board.push(result.move)
                if not no_display:
                    board_status.print(board)
                if not one_round and not automatic:
                    print("one-move:                  'Enter'")
                    print("stop:                      s")
                    print("one-round:                 r")
                    print("automatic with display:    a")
                    print("automatic without display: n")
                    console_input = input("how to proceed ?")
                    if console_input == 's':
                        stop = True
                    elif console_input == 'r':
                        one_round = True
                    elif console_input == 'a':
                        automatic = True
                    elif console_input == 'n':
                        automatic = True
                        no_display = True
                if board.is_game_over():
                    reason = board_status.reason_why_the_game_is_over(board)
                    if not no_display:
                        print("--------------------------------------------")
                        print(reason)
                    if reason == "Termination.CHECKMATE":
                        wins[is_white] += 1
                    else:
                        wins['Draw'] += 1
                    break

            if not stop:
                print(f"wins: white = {wins[True]} - black = {wins[False]} - draw = {wins['Draw']}")
                if not no_display:
                    print("--------------------------------------------")
            else:
                break

    if stop:
        print("stopped")

    end = timer()
    print(f"time: {(end - start):0.1f} sec")


if __name__ == "__main__":
    run()