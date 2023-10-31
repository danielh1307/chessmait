import chess

# <FEN> ::=
# lower case = black, upper case = white
# <Piece Placement>
# ' ' <Side to move>
# ' ' <Castling ability>
# ' ' <En passant target square>
# ' ' <Halfmove clock>
# ' ' <Fullmove counter>
#
# 1abcdefgh/2abcdefgh/3abcdefgh/4abcdefgh/5abcdefgh/6abcdefgh/7abcdefgh/8abcdefgh


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

def is_checkmate(current_position):
    try:
        board = chess.Board(current_position)
        return board.is_checkmate()

    except:
        return False
