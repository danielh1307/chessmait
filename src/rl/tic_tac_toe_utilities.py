import numpy as np


COLOR_WIN = '\033[1;91m'
COLOR_DEFAULT = '\033[0m'


def print_cache(cache, _round, indexes):
    for _i in range(_round):
        print(f"{_i + 1}-- | ", end='')
    print()
    for row in range(3):
        for r in range(_round):
            if r == _round-1:
                print_winner_line(cache[r], row, indexes)
            else:
                print_line(cache[r], row)
        print()


def print_line(line, row):
    for _i in range(3):
        print(f"{line[row * 3 + _i]:.0f}", end='')
    print(" | ", end='')


def print_winner_line(line, row, indexes):
    for _i in range(3):
        if (row * 3 + _i) in indexes and np.sum(indexes) > 0:
            print(f"{COLOR_WIN}{line[row * 3 + _i]:.0f}{COLOR_DEFAULT}", end='')
        else:
            print(f"{line[row * 3 + _i]:.0f}", end='')
    print(" | ", end='')


def cache_board(cache, _round, board):
    cache[_round] = np.array(board.squares)


'''
Unfortunately, the reward from the environment doesn't represent the real outcome of the game.
Therefore, it is manually determined, whether there is a winner
'''
def check_winner(board, winning_combinations):
    for win in np.array(winning_combinations):
        if board[win[0]] != 0 and board[win[0]] == board[win[1]] == board[win[2]]:
            return True, win
    return False, [0, 0, 0]


def print_summary(_games):
    print("----------------------")
    for k in sorted(_games.keys()):
        print(f"{k}: {_games[k]}")
    print("----------------------")


def get_reward(env, agent):
    winner, _ = check_winner(env.board.squares, env.board.winning_combinations)
    if winner and agent == "player_1":
        return 1.0, agent
    elif winner and agent == "player_2":
        return -1.0, agent
    else:
        return 0, "draw"


