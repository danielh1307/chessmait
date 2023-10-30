import os

import chess.pgn


# This class shows how to use python-chess


def test_read_pgn():
    # test that shows how to read a PGN file
    with open(os.path.join("test", "resources", "test-data.pgn")) as pgn_file:
        while True:
            # parse the game
            game = chess.pgn.read_game(pgn_file)

            if game is None:
                break

            print("Moves:")
            print(game.mainline_moves())


def test_pgn_to_fen():
    # test that shows how to convert a PGN to FEN
    with open(os.path.join("test", "resources", "test-data.pgn")) as pgn_file:
        while True:
            # parse the game
            game = chess.pgn.read_game(pgn_file)

            if game is None:
                break

            board = game.board()

            for move in game.mainline_moves():
                board.push(move)

                fen = board.fen()
                print(f"FEN after move {board.fullmove_number}: {fen}")
            print("-" * 40)
