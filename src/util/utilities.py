import chess
import numpy as np
import torch

CLASS_WIN = 1
CLASS_DRAW = 0
CLASS_LOSS = 2


# Check checkmate
def is_checkmate(fen) -> bool:
    """
    Checks if FEN is representing a checkmate position.

    Parameters
    ----------
    fen String input as FEN

    Returns
    -------
    True if checkmate
    """
    board = chess.Board()
    board.set_fen(fen)
    return board.is_checkmate()


# The stalemate is a position where one player has no legal moves available and they are not in check.
def is_stalemate(fen) -> bool:
    """
    Checks if FEN is representing as stalemate.

    Parameters
    ----------
    fen String input as FEN

    Returns
    -------
    true if stalemate
    """
    board = chess.Board()
    board.set_fen(fen)
    return board.is_stalemate()


def fen_to_tensor(fen):
    """
    Converts FEN into a tenor of 384 representing the board.

    Parameters
    ----------
    fen String input as FEN

    Returns
    -------
    Array size  12 (Figures 6xWHITE, 6xBLACK) 8x8 (board). First 12 values represent the figures.
    For each color and type there is an array of 64 fields.
    Values are 1 for turn -1 for wait.
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
    Arrays:
    0:   Pawn, white
    1:   Knight, white
    2:   Bishop, white
    3:   Rook, white
    5:   Queen, white
    6:   King, white
    7:   Pawn, black
    8:   Knight, black
    9:   Bishop, black
    10:  Rook, black
    11:  Queen, black
    12:  Knight, black
    """
    board = chess.Board()
    board.set_fen(fen)

    result = np.zeros((12, 64))
    for sq in chess.SQUARES:
        piece_type = board.piece_type_at(sq)
        if piece_type != 0: # Not no color
            if board.color_at(sq) == chess.WHITE: # white color on layer 1-6
                piece_layer = piece_type
            else: # black color on layer 6-12
                piece_layer = int(piece_type or 0) + 6
            piece_layer = piece_layer - 1
            result[piece_layer, sq] =  (-1 if board.color_at(sq) == chess.WHITE else 1) * (-1 if board.turn == chess.WHITE else 1)
    return result


def main():
    tensor = fen_to_tensor("r1bqnrk1/2p2pbp/p1n1p1p1/1p1pP2P/3P1P2/3BBN2/PPP1N1P1/R2QK2R w")
    print(tensor)

if __name__ == "__main__":
    main()
