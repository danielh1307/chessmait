from pettingzoo.classic import tictactoe_v3
import numpy as np


COLOR_WIN = '\033[1;91m'
COLOR_DEFAULT = '\033[0m'


def run(_env):

    observation, reward, termination, truncation, info = _env.last()

    cache = np.zeros((9, 9))

    rounds = 0

    while not termination and rounds < 10:

        agent = _env.agents[rounds % 2]

        mask = observation["action_mask"]
        # this is where you would insert your policy
        action = _env.action_space(agent).sample(mask)

        _env.step(action)
        observation, reward, termination, truncation, info = _env.last()

        cache_board(cache, rounds, _env.board)
        if termination:
            winner, indexes = check_winner(cache[rounds], _env.board.winning_combinations)
            if winner:
                winner_name = agent
                print(f"{COLOR_WIN}won by: {agent}{COLOR_DEFAULT}")
            else:
                winner_name = "draw"
                print("draw")

        rounds += 1

    print_cache(cache, rounds, indexes)
    return winner_name


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


def check_winner(board, winning_combinations):
    for win in np.array(winning_combinations):
        if board[win[0]] == board[win[1]] == board[win[2]]:
            return True, win
    return False, [0, 0, 0]


if __name__ == "__main__":
    print("RL start ---")
    env = tictactoe_v3.raw_env()
    env.reset(seed=42)
    games = {}
    for i in range(10):
        winner = run(env)
        if winner not in games.keys():
            games[winner] = 0
        games[winner] += 1
        env.reset()
    env.close()
    print(games)
    print("RL end ---")
