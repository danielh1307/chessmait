import chess

# FEN Definition (taken from Wikipedia)
# -------------------------------------
# A FEN record defines a particular game position, all in one text line and using only the ASCII character set.
# It contains six fields, each separated by a space. The fields are as follows:
# 1. Piece placement data
#    - Each rank is described, starting with rank 8 and ending with rank 1, with a "/" between each one (vertical)
#    - Within each rank, the contents of the squares are described in order from the a-file to the h-file (horizontal)
#    - Each piece is identified by a single letter taken from the standard English names:
#      pawn = "P", knight = "N", bishop = "B", rook = "R", queen = "Q" and king = "K"
#    - White pieces are designated using uppercase letters ("PNBRQK"), while black pieces use lowercase letters ("pnbrqk")
#    - A set of one or more consecutive empty squares within a rank is denoted by a digit from "1" to "8", corresponding to the number of squares
# 2. Active color
# 3. Castling availability
# 4. En passant target square
# 5. Halfmove clock
# 6. Fullmove number (starting with 1)

# chessboard layout for FEN:

# file    a b c d e f g h
# rank 1
# rank 2
# rank 3
# rank 4
# rank 5
# rank 6
# rank 7
# rank 8


def get_valid_positions(current_position):
    legal_moves_fen = []
    try:
        board = chess.Board(current_position)
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            # reset
            new_board = chess.Board(current_position)
            new_board.push_uci(move.uci())
            legal_moves_fen.append(new_board.fen())
    finally:
        return legal_moves_fen
