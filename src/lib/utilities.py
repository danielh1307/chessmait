import fnmatch
import os

import chess
import pandas as pd
import torch


def get_device():
    """
    Checks which device is most appropriate to perform the training.
    If cuda is available, cuda is returned, otherwise mps or cpu.

    Returns
    -------
    str
        the device which is used to perform the training.

    """
    _device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"We are going to use {_device} device ...")
    return _device


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
