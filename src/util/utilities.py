import chess
import numpy as np
import torch


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
    Array size 8x8 (board) x6 (Figure w/color). First six values represent the figure. 1 for turn -1 for wait.
    """
    board = chess.Board()
    board.set_fen(fen)
    chess_dict = {
            1: [1,0,0,0,0,0],  # Pawn
            2: [0,1,0,0,0,0],  # Knight
            3: [0,0,1,0,0,0],  # Bishop
            4: [0,0,0,1,0,0],  # Rook
            5: [0,0,0,0,1,0],  # Queen
            6: [0,0,0,0,0,1],  # King
            0: [0,0,0,0,0,0]   #
        }
    return torch.from_numpy(np.array([np.array(chess_dict[(board.piece_type_at(sq) if board.piece_type_at(sq) else 0)])  * (-1 if board.color_at(sq) == False else 1) * (-1 if board.turn == chess.WHITE else 1)for sq in chess.SQUARES]).astype(np.float16).reshape(-1))



def main():
    tensor = fen_to_tensor("r1bqnrk1/2p2pbp/p1n1p1p1/1p1pP2P/3P1P2/3BBN2/PPP1N1P1/R2QK2R b")
    print(tensor)

if __name__ == "__main__":
    main()
