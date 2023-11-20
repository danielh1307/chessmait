
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

    def convert_stockfish(self, board):
        board = (board.replace("   ",".").replace("+", "").replace("-", "").replace("|","").replace(" ", "")
                 .replace("1","").replace("2","").replace("3","").replace("4","")
                 .replace("5","").replace("6","").replace("7","").replace("8","")
                 .replace("a.b.c.d.e.f.g.h","")
                 .split())
        while "" in board:
            board.remove("")
        return board

    def cache_stockfish(self, board):
        self.old_board = self.convert_stockfish(board)

    def print_stockfish(self, new_board):
        new_board = self.convert_stockfish(new_board)
        for row in range(8):
            for col in range(8):
                if new_board[row][col] != self.old_board[row][col]:
                    print(f"{BoardStatus.COLOR_MOVES}{new_board[row][col]}{BoardStatus.COLOR_DEFAULT} ", end='')
                else:
                    print(f"{new_board[row][col]} ", end='')
            print("")

    def reason_why_the_game_is_over(self, board, max_rounds_reached):
        outcome = board.outcome()
        if outcome == None:
            return "no outcome"
        reason = "winner: "
        if outcome.winner == None:
            reason += "None"
        else:
            reason += "black" if not outcome.winner else "white"
        if max_rounds_reached:
            reason += " --- outcome: max rounds reached"
        else:
            reason += " --- outcome: " + outcome.result()
        return reason


