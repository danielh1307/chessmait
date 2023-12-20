import os
from timeit import default_timer as timer

import chess.engine

import play.board_status as bs
import play.chess_engine as ce
import play.trained_model as tm

PATH_TO_CHESS_ENGINE = os.path.join("stockfish", "stockfish-windows-x86-64-avx2.exe")
NUMBER_OF_GAMES_PER_POSITION = 1

starting_positions = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Initial position"),
    ("rnbqkbnr/pppp1ppp/8/8/4Pp2/8/PPPP2PP/RNBQKBNR w KQkq - 0 3", "King's Gambit accepted"),
    ("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", "Spanish opening"),
    ("rnbqkbnr/pp1ppppp/8/8/3pP3/2P5/PP3PPP/RNBQKBNR b KQkq - 0 3", "Smith Morra Gambit"),
    ("r1bqk1nr/pppp1ppp/2n5/2b1p3/1PB1P3/5N2/P1PP1PPP/RNBQK2R b KQkq b3 0 4", "Evan's Gambit")
]

all_moves = []
all_fen = []


def run():
    board_status = bs.BoardStatus()

    # set one to None to let Stockfish play
    model_white = tm.model_wild_snow_28
    model_black = tm.model_apricot_armadillo_167

    start = timer()
    no_display = False

    wins = {True: 0, False: 0, 'Draw': 0, '+1000': 0, '-1000': 0}
    with chess.engine.SimpleEngine.popen_uci(PATH_TO_CHESS_ENGINE) as stockfish_engine:
        stockfish_engine.configure({"UCI_Elo": 1320})

        automatic = False
        stop = False
        for start_position_fen, start_position_name in starting_positions:
            print("***************************************")
            print("New starting position: ", start_position_name)
            print("***************************************")
            for i in range(1, NUMBER_OF_GAMES_PER_POSITION + 1):
                max_score_reached = False
                board = chess.Board()
                board.set_fen(start_position_fen)
                all_moves.clear()
                all_fen.clear()
                all_fen.append(start_position_fen)
                one_round = False
                stop = False

                moves = 0
                if not no_display:
                    print(f"Round: {i}")
                while not stop:
                    moves += 1
                    white_to_move = board.turn == chess.WHITE
                    board_status.cache(board)
                    model_to_play = model_white if white_to_move else model_black
                    if not no_display:
                        print(f"--------------- Round: {i} --- Moves: {moves:02d}")
                        print(f"White: {model_white.get_name()}")
                        print(f"Black: {model_black.get_name()}")
                        print(f"{'White to move' if white_to_move else 'Black to move'}")
                        print(board)

                    score = stockfish_engine.analyse(board, chess.engine.Limit(depth=10))["score"]
                    score = score.white() if score.white().score() is None else score.white().score()

                    if not one_round and not automatic:
                        print("one-move:                  'Enter'")
                        print("stop:                      s")
                        print("one-round:                 r")
                        print("automatic with display:    a")
                        print("automatic without display: n")
                        # print(str(game))
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
                    max_reached_now = type(score) != int or score > 1000 or score < -1000
                    if not max_score_reached and max_reached_now:
                        max_score_reached = True
                        print(f"Score {score} is reached")
                        if type(score) == int:
                            if score > 1000:
                                wins['+1000'] += 1
                            else:
                                wins['-1000'] += 1
                        else:
                            if score.mate() > 0:
                                wins['+1000'] += 1
                            else:
                                wins['-1000'] += 1
                    if board.is_game_over(claim_draw=True):
                        reason = board_status.reason_why_the_game_is_over(board)
                        if reason == "Termination.CHECKMATE":
                            wins[not white_to_move] += 1
                        else:
                            wins['Draw'] += 1
                        print("--------------------------------------------")
                        print(reason)
                        break

                    play(board, model_to_play, white_to_move)

                if not stop:
                    print(f"White wins: {wins[True]}")
                    print(f"Black wins: {wins[False]}")
                    print(f"Draws: {wins['Draw']}")
                    print(f"White reaches +1000 first: {wins['+1000']}")
                    print(f"Black reaches -1000 first: {wins['-1000']}")
                    for move in all_moves:
                        print(move, end=' ')
                else:
                    break
    if stop:
        print("stopped")

    end = timer()
    print(f"time: {(end - start):0.1f} sec")


def play(board, model_to_play, is_white):
    # Model to move
    move, fen = ce.play_return_move(board, model_to_play, is_white, all_fen)
    all_fen.append(fen)
    all_moves.append(move)


if __name__ == "__main__":
    run()
