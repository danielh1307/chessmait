import chess


# Check checkmate
def is_checkmate(fen) -> bool:
    board = chess.Board()
    board.set_fen(fen)
    return board.is_checkmate()


def is_stalemate(fen) -> bool:
    board = chess.Board()
    board.set_fen(fen)
    return board.is_stalemate()
