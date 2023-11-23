import chess
import torch
import os
import fnmatch

import pandas as pd

CLASS_WIN = 1
CLASS_DRAW = 0
CLASS_LOSS = 2


def get_files_from_pattern(directory, file_pattern):
    """

    Parameters
    ----------
    directory : str
        a directory
    file_pattern : str
        a file name pattern

    Returns
    -------
    list
        Returns all files (including directory) matching the pattern in the given directory

    """
    matching_files = [file for file in os.listdir(directory) if fnmatch.fnmatch(file, file_pattern)]
    return [os.path.join(directory, file) for file in matching_files]


def dataframe_from_files(file_names_with_directory, pickle_files=False):
    """

    Parameters
    ----------
    directory : str
        a directory
    file_names_with_directory : list
        a list of file names


    Returns
    -------
    pd.Dataframe
        a single Dataframe of all the given files in the given directory without the index

    """
    _dataframes = []
    for file_name_with_directory in file_names_with_directory:
        print(f"Read {file_name_with_directory} ...")
        if pickle_files:
            _dataframes.append(pd.read_pickle(file_name_with_directory))
        else:
            _dataframes.append(pd.read_csv(file_name_with_directory))

    return pd.concat(_dataframes, ignore_index=True)


def write_values_in_bars(curr_plot):
    """
    Writes the values from bar plots into the bars.
    """
    for p in curr_plot.patches:
        curr_plot.annotate(format(p.get_height(), '.1f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           xytext=(0, 9),
                           textcoords='offset points')


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


def fen_to_cnn_tensor(fen):
    """
    Converts FEN into a tensor of 1x12x8x8 dimensions representing the board. Figures x H x W (C, H, W according to
    Conv2d).

    Parameters
    ----------
    fen String input as FEN

    Returns
    -------
    Array size  12 (Figures 6xWHITE, 6xBLACK) x 8 (Height) x 8 (Weight).
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

    result = torch.zeros((12, 8, 8))
    for sq in chess.SQUARES:
        piece_type = board.piece_type_at(sq)
        if piece_type != 0:  # Not no color
            if board.color_at(sq) == chess.WHITE:  # white color on layer 1-6
                piece_layer = piece_type
            else:  # black color on layer 6-12
                piece_layer = int(piece_type or 0) + 6
            piece_layer = piece_layer - 1
            result[piece_layer, int(sq / 8), sq % 8] = (-1 if board.color_at(sq) == chess.WHITE else 1) * \
                                                     (1 if board.turn == chess.WHITE else -1)
    return result


def fen_to_cnn_tensor_non_hot_enc(fen):
    """
    Converts FEN into a tensor of 8x8 dimensions representing the board. 1 x H x W (C, H, W according to Conv2d).

    Parameters
    ----------
    fen String input as FEN

    Returns
    -------
    Array size 8 (Height) x 8 (Weight).
    Values are positive for turn and negative for wait.
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING

    Figures:
    0:   No figure
    1:   Pawn
    2:   Knight
    3:   Bishop
    4:   Rook
    5:   Queen
    6:   King
    """
    # Define the piece types
    pieces = 'pnbrqkPNBRQK'
    piece_to_index = {
        'p': 1,
        'n': 3,
        'b': 4,
        'r': 5,
        'q': 9,
        'k': 10,
        'P': 1,
        'N': 3,
        'B': 4,
        'R': 5,
        'Q': 9,
        'K': 10
    }

    # Initialize an empty board
    board = torch.zeros(2, 8, 8)

    # Split the FEN string to get the board layout and current player
    position, player = fen.split(' ')[0:2]

    # Replace numbers with the corresponding number of empty squares
    for i in range(1, 9):
        position = position.replace(str(i), '.' * i)

    # Replace slashes with empty spaces
    position = position.replace('/', '')

    # Fill the board tensor
    for i, piece in enumerate(position):
        row = i // 8
        col = i % 8
        if piece in piece_to_index:
            if piece.isupper():
                layer = 1
            else:
                layer = 0
            # Calculate the piece's binary value (1 for current player, -1 for opponent)
            if (player == 'w' and piece.isupper()) or (player == 'b' and not piece.isupper()):
                multiplicator = 1
            else:
                multiplicator = -1
            board[layer, row, col] = piece_to_index[piece] * multiplicator
            # board[row, [col, piece_to_index[piece]]] = value

    # Reshape the tensor to the desired shape (768,)
    return board


def fen_to_tensor_non_hot_enc_1dim(fen):
    """
    Converts FEN into a tensor of 1x64 dimensions representing the board.

    Parameters
    ----------
    fen String input as FEN

    Returns
    -------
    Array size 1 (Height) x 64 (Weight).

    """
    piece_to_index = {
        'p': -1,
        'r': -3,
        'n': -4,
        'b': -6,
        'q': -8,
        'k': -10,
        'P': 1,
        'R': 3,
        'N': 4,
        'B': 6,
        'Q': 8,
        'K': 10
    }

    # Initialize an empty board
    board = torch.zeros(64)

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
            board[i] = piece_to_index[piece]

    # Reshape the tensor to the desired shape (64,)
    return board


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


def fen_to_tensor_one_board_dense(fen):
    """
    Converts FEN into a tensor of 64x2 dimensions representing the board.

    """

    # Define the piece values
    pieces = {
        "p": 1,
        "n": 3,
        "b": 4,
        "r": 5,
        "q": 9,
        "k": 10
    }

    # Initialize an empty board
    board = torch.zeros(64, 2)

    # Split the FEN string to get the board layout and current player
    position, player = fen.split(' ')[0:2]

    # Replace numbers with the corresponding number of empty squares
    for i in range(1, 9):
        position = position.replace(str(i), '.' * i)

    # Replace slashes with empty spaces
    position = position.replace('/', '')

    # Fill the board tensor
    for i, piece in enumerate(position):
        if piece.lower() in pieces.keys():
            # Calculate the piece's binary value (1 for current player, -1 for opponent)
            if (player == 'w' and piece.isupper()) or (player == 'b' and not piece.isupper()):
                value = pieces.get(piece.lower())
            else:
                value = -1 * pieces.get(piece.lower())
            piece_index = 0 if piece.isupper() else 1
            board[i, piece_index] = value

    # Reshape the tensor to the desired shape (768,)
    return board.view(-1)


def fen_to_cnn_tensor_alternative(fen):
    # Define the piece types
    pieces = 'pnbrqkPNBRQK'
    piece_to_index = {piece: index for index, piece in enumerate(pieces)}

    # Initialize an empty board
    board = torch.zeros(12, 8, 8)

    # Split the FEN string to get the board layout and current player
    position, player = fen.split(' ')[0:2]

    # Replace numbers with the corresponding number of empty squares
    for i in range(1, 9):
        position = position.replace(str(i), '.' * i)

    # Replace slashes with empty spaces
    position = position.replace('/', '')

    # Fill the board tensor
    for i, piece in enumerate(position):
        row = i // 8
        col = i % 8
        if piece in piece_to_index:
            # Calculate the piece's binary value (1 for current player, -1 for opponent)
            if (player == 'w' and piece.isupper()) or (player == 'b' and not piece.isupper()):
                value = 1
            else:
                value = -1
            board[piece_to_index[piece], row, col] = value
            # board[row, [col, piece_to_index[piece]]] = value

    # Reshape the tensor to the desired shape (768,)
    return board


def main():
    tensor = fen_to_tensor_one_board_dense("r1bqnrk1/2p2pbp/p1n1p1p1/1p1pP2P/3P1P2/3BBN2/PPP1N1P1/R2QK2R w")
    print(tensor.shape)
    print(tensor)
    tensor = fen_to_tensor_non_hot_enc_1dim("r1bqnrk1/2p2pbp/p1n1p1p1/1p1pP2P/3P1P2/3BBN2/PPP1N1P1/R2QK2R w")
    print(tensor)
    tensor = fen_to_tensor_non_hot_enc_1dim("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print(tensor)


if __name__ == "__main__":
    main()
