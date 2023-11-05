import chess


# Check checkmate
def is_checkmate(fen) -> bool:
    board = chess.Board()
    board.set_fen(fen)
    return board.is_checkmate()


# The stalemate is a position where one player has no legal moves available and they are not in check.
def is_stalemate(fen) -> bool:
    board = chess.Board()
    board.set_fen(fen)
    return board.is_stalemate()
