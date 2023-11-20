import os
import src.board_status as bs
from stockfish import Stockfish

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
    board_status = bs.BoardStatus()

    stop = False
    do_not_stop = False
    round = 0
    max_rounds = 10

    stockfish = Stockfish(path=PATH_TO_CHESS_ENGINE)
    #stockfish.set_elo_rating(500)
    stockfish.set_skill_level(20)
    #stockfish.set_depth()

    while not stop:
        round += 1
        is_white = (round % 2) == 1
        board_status.cache_stockfish(stockfish.get_board_visual())
        print(f"--------------- Round: {round:02d} {'white' if is_white else 'black'}", end='')
        wdl = stockfish.get_wdl_stats()
        if wdl == None:
            print("")
        else:
            print(f" --- win: {wdl[0]} --- draw: {wdl[1]} --- loss: {wdl[2]}")
        best_move = stockfish.get_best_move()
        if best_move == None:
            message = "game over: winner is " + ("black" if is_white else "white")
            print(message)
            break
        stockfish.make_moves_from_current_position([best_move])
        board_status.print_stockfish(stockfish.get_board_visual())
        if not do_not_stop:
            console_input = input("continue/stop/automatic/max-rounds ('Enter'/s/a/#)? ")
            if console_input == 's':
                stop = True
            elif console_input == 'a':
                do_not_stop = True
            elif console_input != "" and console_input.isdigit():
                max_rounds = int(console_input)
                do_not_stop = True
        if round > max_rounds:
            stop = True


if __name__ == "__main__":
    run()