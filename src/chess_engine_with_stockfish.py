import chess.engine
import os
import src.board_status as bs

PATH_TO_CHESS_ENGINE = os.path.join("stockfish", "stockfish-windows-x86-64-avx2.exe")

def run():
    board = chess.Board()
    board_status = bs.BoardStatus()

    stop = False
    do_not_stop = False
    round = 0
    max_rounds = 10

    with chess.engine.SimpleEngine.popen_uci(PATH_TO_CHESS_ENGINE) as engine:
        engine.configure({"UCI_Elo": 1320})

        while not stop:
            round += 1
            is_white = (round % 2) == 1
            board_status.cache(board)
            print(f"--------------- Round: {round:02d} {'white' if is_white else 'black'}")
            result = engine.play(board, chess.engine.Limit(time=0.01, depth=None, nodes=None))
            opt = engine.options.keys()
            board.push(result.move)
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
            if board.is_game_over() or round > max_rounds:
                stop = True

    print("--------------------------------------------")
    print(board_status.reason_why_the_game_is_over(board, round > max_rounds))
    print("--------------------------------------------")


if __name__ == "__main__":
    run()