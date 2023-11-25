# inspired by https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542

from pettingzoo.classic import tictactoe_v3
import numpy as np


COLOR_WIN = '\033[1;91m'
COLOR_DEFAULT = '\033[0m'

e_greedy = 0.3
learning_rate = 0.2
gamma_decay = 0.9

possible_indices = np.arange(9)


def run(_env, _q_table):

    observation, reward, termination, truncation, info = _env.last()

    cache = np.zeros((9, 9))

    rounds = 0

    state_table = { _env.agents[0]: [], _env.agents[1]: []}

    while not termination and rounds < 10:

        agent = _env.agents[rounds % 2]
        other_agent = _env.agents[np.abs((rounds % 2)-1)]

        mask = observation["action_mask"]
        allowed_indices = possible_indices[mask == 1]
        next_position_index = -1
        if np.random.uniform(0, 1) < e_greedy:
            # set the mask to zeroes, except one randomly selected, possible, index
            next_position_index = np.random.choice(len(allowed_indices))
        else:
            max_val = np.finfo(float).min
            for idx in range(len(allowed_indices)):
                next_board = np.array(_env.board.squares)
                next_board[allowed_indices[idx]] = 1 if agent == 'player_1' else 2
                next_board_hash = str(next_board)
                value = 0 if _q_table[agent].get(next_board_hash) is None else _q_table[agent].get(next_board_hash)
                if value > max_val:
                    max_val = value
                    next_position_index = idx
        mask = np.zeros(9, dtype=np.int8)
        mask[allowed_indices[next_position_index]] = 1

        action = _env.action_space(agent).sample(mask)

        _env.step(action)
        state_table[agent].append(np.array(_env.board.squares))
        observation, reward, termination, truncation, info = _env.last()

        cache_board(cache, rounds, _env.board)
        if termination:
            winner, indexes = check_winner(cache[rounds], _env.board.winning_combinations)
            if winner:
                give_reward(_q_table[agent], state_table[agent], 1.0)
                give_reward(_q_table[other_agent], state_table[other_agent], 0.0)
                winner_name = agent
                print(f"{COLOR_WIN}won by: {agent}{COLOR_DEFAULT}")
            else:
                give_reward(_q_table[agent], state_table[agent], 0.0)
                give_reward(_q_table[other_agent], state_table[other_agent], 0.0)
                winner_name = "draw"
                print("draw")

        rounds += 1

    print_cache(cache, rounds, indexes)
    return winner_name


def give_reward(_q_table, states, reward):
    for state in reversed(states):
        state_hash = str(state)
        if _q_table.get(state_hash) is None:
            _q_table[state_hash] = 0.0
        _q_table[state_hash] += learning_rate * (gamma_decay * reward - _q_table[state_hash])
        reward = _q_table[state_hash]


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
    q_table = {env.agents[0]: {}, env.agents[1]: {}}
    games = {}
    for i in range(100):
        winner = run(env, q_table)
        if winner not in games.keys():
            games[winner] = 0
        games[winner] += 1
        env.reset()
    env.close()
    print(games)
    print("RL end ---")
