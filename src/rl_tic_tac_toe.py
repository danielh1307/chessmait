from pettingzoo.classic import tictactoe_v3
import pettingzoo.utils as utils
import numpy as np


def run(env):

    observation, reward, termination, truncation, info = env.last()

    total_reward = 0.0

    cache = np.zeros((9,3,3))

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
            if check_winner(cache[round]):
                print(f"won by: {agent}")
            else:
                print("draw")

        round += 1

    print_cache(cache, round)


def print_cache(cache, round):
    for i in range(round):
        print(f"{i+1}-- | ", end='')
    print()
    for i in range(3):
        for j in range(round):
            print_line(cache[j][i])
        print()


def print_line(line):
    print(f"{line[0]:.0f}{line[1]:.0f}{line[2]:.0f} | ", end='')


def cache_board(cache, round, board):
    board = np.array(board.squares)
    cache[round][0] = board[0:3]
    cache[round][1] = board[3:6]
    cache[round][2] = board[6:9]


def check_winner(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2]: # horizontal
            return True
        if board[0][i] == board[1][i] == board[2][i]: # vertical
            return True
    if board[0][0] == board[1][1] == board[2][2]:  # diagonal left/upper to right/lower
        return True
    if board[0][2] == board[1][1] == board[2][0]:  # diagonal left/lower to right/upper
        return True
    return False

if __name__ == "__main__":
    print("RL start ---")
    env = tictactoe_v3.raw_env()
    env.reset(seed=42)
    for i in range(10):
        run(env)
        env.reset()

    env.close()
    print("RL end ---")
