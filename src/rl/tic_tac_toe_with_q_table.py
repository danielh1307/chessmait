# inspired by https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542

'''
Für Tic-Tac-Toe gibt es 255.168 verschiedene Spielverläufe, von denen 131.184 mit einem Sieg des
ersten Spielers enden, 77.904 mit einem Sieg des zweiten Spielers und 46.080 mit einem Unentschieden
'''
import os
import sys
from pettingzoo.classic import tictactoe_v3
from src.rl.tic_tac_toe_utilities import *
import csv
import time

GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.3
LR = 0.1

POSSIBLE_INDEXES = np.arange(9)
MAX_NUMBER_OF_ITERATIONS = 1_000_001


def run(_env, _q_table, training, show, num_of_iteration):

    observation, reward, termination, truncation, info = _env.last()

    rounds = 0

    state_table = []
    cache = np.zeros((9, 9))

    indexes = [0, 0, 0]
    winner_name = ""

    while not termination and rounds < 10:

        agent = _env.agents[rounds % 2]

        mask = observation["action_mask"]
        action = select_action(_env, _q_table, mask, agent, training, num_of_iteration)

        _env.step(action)
        if training and agent == "player_1":
            state_table.append(np.array(_env.board.squares))
        observation, reward, termination, truncation, info = _env.last()

        cache_board(cache, rounds, _env.board)
        if termination:
            _winner, indexes = check_winner(cache[rounds], _env.board.winning_combinations)
            if _winner:
                winner_name = agent
                if training and agent == "player_1":
                    give_reward(_q_table, state_table, 1.0)
                elif show:
                    print(f"{COLOR_WIN}won by: {agent}{COLOR_DEFAULT}")
            else:
                winner_name = "draw"
                if training and agent == "player_1":
                    give_reward(_q_table, state_table, 0.0)
                elif show:
                    print("draw")

        rounds += 1

    if not training and show:
        print_cache(cache, rounds, indexes)
    return winner_name


def select_action(_env, _q_table, _mask, _agent, _training, num_of_iteration):
    sample = np.random.random()
    eps_threshold = EPS_START - (EPS_START - EPS_END) * num_of_iteration / MAX_NUMBER_OF_ITERATIONS
    if _agent == "player_1" and eps_threshold > sample and _training:
        return _env.action_space(_agent).sample(_mask)
    elif _agent == "player_2":
        return _env.action_space(_agent).sample(_mask)
    else:
        allowed_indices = POSSIBLE_INDEXES[_mask == 1]
        max_val = np.finfo(float).min
        next_position_index = 0
        for idx in range(len(allowed_indices)):
            next_board = np.array(_env.board.squares)
            next_board[allowed_indices[idx]] = 1 if _agent == 'player_1' else 2
            next_board_hash = str(next_board)
            value = 0 if _q_table.get(next_board_hash) is None else _q_table.get(next_board_hash)
            if value > max_val:
                max_val = value
                next_position_index = idx
        return allowed_indices[next_position_index]


def give_reward(_q_table, states, reward):
    for state in reversed(states):
        state_hash = str(state)
        if _q_table.get(state_hash) is None:
            _q_table[state_hash] = 0.0
        _q_table[state_hash] += LR * (GAMMA * reward - _q_table[state_hash])
        reward = _q_table[state_hash]


def get_size_of_q_table(_q_table):
    size_in_in_bytes = sys.getsizeof(q_table)
    for k, v in _q_table.items():
        size_in_in_bytes += sys.getsizeof(k)
        size_in_in_bytes += sys.getsizeof(v)
    return size_in_in_bytes


if __name__ == "__main__":
    start = time.time()
    file_name = "tic-tac-toe-statistics-q-table.csv"
    env = tictactoe_v3.raw_env()
    env.reset(seed=42)
    q_table = {}
    if os.path.isfile(file_name):
        os.remove(file_name)
    with open(file_name, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["#of-trainings\tp1-training-won\t#p2-training-won\tdraw-training\t#of-test-games\tp1-test-won\tp2-test-won\tdraw-test\tq-table-size"])
    games_training = {"draw": 0, "player_1": 0, "player_2": 0}
    print("RL training start ***")
    number_of_iterations = 0
    for i in range(MAX_NUMBER_OF_ITERATIONS):
        number_of_iterations += 1
        winner = run(env, q_table, True, False, i)
        games_training[winner] += 1
        env.reset()
        if (number_of_iterations-1) in [10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 2000, 4000, 6000, 8000,
                                        10_000, 20_000, 40_000, 60_000, 80_000, 100_000, 200_000, 400_000, 600_000,
                                        800_000, 1_000_000]:
            print(f"iteration: {i}")
            print_summary(games_training)
            print("RL test start ***")
            games_test = {"draw": 0, "player_1": 0, "player_2": 0}
            for j in range(1000):
                winner = run(env, q_table, False, False, i)
                games_test[winner] += 1
                env.reset()
            print_summary(games_test)
            with open(file_name, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([f"{i}\t{games_training['player_1']}\t{games_training['player_2']}\t{games_training['draw']}\t1000\t{games_test['player_1']}\t{games_test['player_2']}\t{games_test['draw']}\t{len(q_table)}"])
            print("RL test end ***")
    print("RL training end ***")
    env.close()
    end = time.time()
    print(f"duration: {(end - start):0.1f}s")
