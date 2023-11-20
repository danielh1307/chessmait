
class BoardStatus:
    # colors and font settings see: https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
    COLOR_MOVES = '\033[1;91m'
    COLOR_EMPTY_FIELDS = '\033[37m'
    COLOR_DEFAULT = '\033[0m'

    int_to_str_mapping = {
        1: 'P',  # White Pawn
        -1: 'p',  # Black Pawn
        2: 'N',  # White Knight
        -2: 'n',  # Black Knight
        3: 'B',  # White Bishop
        -3: 'b',  # Black Bishop
        4: 'R',  # White Rook
        -4: 'r',  # Black Rook
        5: 'Q',  # White Queen
        -5: 'q',  # Black Queen
        6: 'K',  # White King
        -6: 'k',  # Black King
        0: '.',
    }

    def cache(self, board):
        self.old_board = self.convert_to_int(board)


    def convert_to_int(self, board):
        indices = '♚♛♜♝♞♟⭘♙♘♗♖♕♔'
        unicode = board.unicode()
        return [
            [indices.index(c)-6 for c in row.split()]
            for row in unicode.split('\n')
        ]


    def print(self, new_board):
        new_board = self.convert_to_int(new_board)
        for row in range(8):
            for col in range(8):
                if new_board[row][col] != self.old_board[row][col]:
                    print(f"{BoardStatus.COLOR_MOVES}{BoardStatus.int_to_str_mapping[new_board[row][col]]}{BoardStatus.COLOR_DEFAULT} ", end='')
                elif BoardStatus.int_to_str_mapping[new_board[row][col]] == '.':
                    print(f"{BoardStatus.COLOR_EMPTY_FIELDS}{BoardStatus.int_to_str_mapping[new_board[row][col]]}{BoardStatus.COLOR_DEFAULT} ", end='')
                else:
                    print(f"{BoardStatus.int_to_str_mapping[new_board[row][col]]} ", end='')
            print("")


    @staticmethod
    def reason_why_the_game_is_over(board):
        return str(board.outcome().termination)
