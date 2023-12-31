import chess
import numpy as np
import torch


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


def fen_to_tensor_simple(fen):
    """

    Parameters
    ----------
    fen

    Returns
    -------
    8x8 board, 6x position, 2x color, draw (one value) 1 for white, 0 for black

    """
    # Define the piece types
    pieces = 'pnbrqkPNBRQK'
    piece_to_index = {
        'p': 1,
        'n': 2,
        'b': 3,
        'r': 4,
        'q': 5,
        'k': 6,
        'P': 1,
        'N': 2,
        'B': 3,
        'R': 4,
        'Q': 5,
        'K': 6
    }

    # Initialize an empty board
    board = torch.zeros(8 * 8 * 6 * 2 + 1)

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
            row = i // 8
            col = i % 8
            # print(f'{row} {col} {piece_to_index[piece]} {piece} {(piece_to_index[piece] + col * 6 + row * 6 * 8) + (8*8*6 if piece.isupper() else 0)}')
            board[((piece_to_index[piece] + col * 6 + row * 6 * 8) + (8 * 8 * 6 if piece.isupper() else 0)) - 1] = 1
    board[8 * 8 * 6 * 2] = 1 if player == 'w' else 0

    # Reshape the tensor to the desired shape (768,)
    return board


def fen_to_bitboard(fen):
    board = chess.Board()
    board.set_fen(fen)

    # dictionary to store bitboards
    piece_bitboards = {}

    # for each color (white, black)
    for color in chess.COLORS:

        # for each piece type (pawn, bishop, knigh, rook, queen, kinb)
        for piece_type in chess.PIECE_TYPES:
            v = board.pieces_mask(piece_type, color)
            symbol = chess.piece_symbol(piece_type)
            i = symbol.upper() if color else symbol
            piece_bitboards[i] = v

    # empty bitboard
    piece_bitboards['-'] = board.occupied ^ 2 ** 64 - 1

    # player bitboard (full 1s if player is white, full 0s otherwise)
    player = 2 ** 64 - 1 if board.turn else 0

    # castling_rights bitboard
    castling_rights = board.castling_rights

    # en passant bitboard
    en_passant = 0
    ep = board.ep_square
    if ep is not None:
        en_passant |= (1 << ep)

    # bitboards (16) = 12 for pieces, 1 for empty squares, 1 for player, 1 for castling rights, 1 for en passant
    bitboards = [b for b in piece_bitboards.values()] + [player] + [castling_rights] + [en_passant]

    # for each bitboard transform integer into a matrix of 1s and 0s
    # reshape in 3D format (16 x 8 x 8)
    bitarray = np.array([
        np.array([(bitboard >> i & 1) for i in range(64)])
        for bitboard in bitboards
    ]).reshape((16, 8, 8))

    return torch.tensor(bitarray, dtype=torch.float32)
