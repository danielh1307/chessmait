from pettingzoo.classic import tictactoe_v3
import pettingzoo.utils as utils
import numpy as np


COLOR_WIN = '\033[1;91m'
COLOR_DEFAULT = '\033[0m'


def run(env):

    observation, reward, termination, truncation, info = env.last()

    total_reward = 0.0

    cache = np.zeros((9,9))

    round = 0

    while not termination and round < 10:

        agent = env.agents[round % 2]

        mask = observation["action_mask"]
        # this is where you would insert your policy
        action = env.action_space(agent).sample(mask)

        env.step(action)
        observation, reward, termination, truncation, info = env.last()

        cache_board(cache, round, env.board)
        if termination:
            winner, indexes = check_winner(cache[round])
            if winner:
                print(f"{COLOR_WIN}won by: {agent}{COLOR_DEFAULT}")
            else:
                print("draw")

        round += 1

    print_cache(cache, round, indexes)


def print_cache(cache, round, indexes):
    for i in range(round):
        print(f"{i+1}-- | ", end='')
    print()
    for row in range(3):
        for r in range(round):
            if r == round-1:
                print_winner_line(cache[r], row, indexes)
            else:
                print_line(cache[r], row)
        print()


def print_line(line, row):
    for i in range(3):
        print(f"{line[row*3+i]:.0f}", end='')
    print(" | ", end='')


def print_winner_line(line, row, indexes):
    for i in range(3):
        if (row*3+i) in indexes and indexes != (0,0,0):
            print(f"{COLOR_WIN}{line[row*3+i]:.0f}{COLOR_DEFAULT}", end='')
        else:
            print(f"{line[row*3+i]:.0f}", end='')
    print(" | ", end='')


def cache_board(cache, round, board):
    cache[round] = np.array(board.squares)


def check_winner(board):
    x = np.resize(board, (3,3))
    for i in range(3):
        if board[i*3] == board[i*3+1] == board[i*3+2] > 0: # horizontal
            return True, (i*3, i*3+1, i*3+2)
        if board[i] == board[i+3] == board[i+6]: # vertical
            return True, (i, i+3, i+6)
    if board[0] == board[4] == board[8]:  # diagonal left/upper to right/lower
        return True, (0, 4, 8)
    if board[6] == board[4] == board[2]:  # diagonal left/lower to right/upper
        return True, (6, 4, 2)
    return False, (0, 0, 0)

if __name__ == "__main__":
    print("RL start ---")
    env = tictactoe_v3.raw_env()
    env.reset(seed=42)
    for i in range(10):
        run(env)
        env.reset()

    env.close()
    print("RL end ---")
