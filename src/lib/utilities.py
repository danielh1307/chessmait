import chess
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
    Converts FEN into a tensor of 12x64 dimensions representing the board.

    Parameters
    ----------
    fen String input as FEN

    Returns
    -------
    Array size  12 (Figures 6xWHITE, 6xBLACK) 8x8 (board). First 12 values represent the figures.
    For each color and type, there is an array of 64 fields.
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

    result = torch.zeros((12, 64))
    for sq in chess.SQUARES:
        piece_type = board.piece_type_at(sq)
        if piece_type != 0:  # Not no color
            if board.color_at(sq) == chess.WHITE:  # white color on layer 1-6
                piece_layer = piece_type
            else:  # black color on layer 6-12
                piece_layer = int(piece_type or 0) + 6
            piece_layer = piece_layer - 1
            result[piece_layer, sq] = (-1 if board.color_at(sq) == chess.WHITE else 1) * \
                                      (-1 if board.turn == chess.WHITE else 1)
    return result


def fen_to_tensor_square_board(fen):
    """
    Converts FEN into a tensor of 8x8x12 dimensions representing the board. H x W x Figures

    Parameters
    ----------
    fen String input as FEN

    Returns
    -------
    Array size  8 (Height) x 8 (Weight) x 12 (Figures 6xWHITE, 6xBLACK).
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

    result = torch.zeros((8, 8, 12))
    for sq in chess.SQUARES:
        piece_type = board.piece_type_at(sq)
        if piece_type != 0:  # Not no color
            if board.color_at(sq) == chess.WHITE:  # white color on layer 1-6
                piece_layer = piece_type
            else:  # black color on layer 6-12
                piece_layer = int(piece_type or 0) + 6
            piece_layer = piece_layer - 1
            result[int(sq / 8), sq % 8, piece_layer] = (-1 if board.color_at(sq) == chess.WHITE else 1) * \
                                       (-1 if board.turn == chess.WHITE else 1)
    return result


def fen_to_tensor_one_board(fen):
    """
    Converts FEN into a tensor of 64x12 dimensions representing the board.

    While fen_to_tensor creates 12 different board representations (one for each piece type), this
    resulting tensor creates just one board representation. For each field, the piece is encoded, so we have
    64 fields and on each field 12 dimensions which encode the piece.
    """

    # Define the piece types
    pieces = 'pnbrqkPNBRQK'
    piece_to_index = {piece: index for index, piece in enumerate(pieces)}

    # Initialize an empty board
    board = torch.zeros(64, 12)

    # Split the FEN string to get the board layout and current player
    position, player = fen.split(' ')[0:2]

    # Replace numbers with the corresponding number of empty squares
    for i in range(1, 9):
        position = position.replace(str(i), '.' * i)

    # Replace slashes with empty spaces
    position = position.replace('/', '')

    # Fill the board tensor
    for i, piece in enumerate(position):
        if piece in piece_to_index:
            # Calculate the piece's binary value (1 for current player, -1 for opponent)
            if (player == 'w' and piece.isupper()) or (player == 'b' and not piece.isupper()):
                value = 1
            else:
                value = -1
            board[i, piece_to_index[piece]] = value

    # Reshape the tensor to the desired shape (768,)
    return board.view(-1)


def main():
    tensor = fen_to_tensor("r1bqnrk1/2p2pbp/p1n1p1p1/1p1pP2P/3P1P2/3BBN2/PPP1N1P1/R2QK2R w")
    print(tensor)


if __name__ == "__main__":
    main()