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

learning_rate = 0.2
gamma_decay = 0.9

possible_indices = np.arange(9)


def run(_env, _q_table, _e_greedy, training, show):

    observation, reward, termination, truncation, info = _env.last()

    rounds = 0

    state_table = { _env.agents[0]: [], _env.agents[1]: []}
    cache = np.zeros((9, 9))

    indexes = [0, 0, 0]
    winner_name = ""

    while not termination and rounds < 10:

        agent = _env.agents[rounds % 2]
        other_agent = _env.agents[np.abs((rounds % 2)-1)]

        mask = observation["action_mask"]
        if np.random.uniform(0, 1) < _e_greedy or not training and agent == 'player_2':
            action = _env.action_space(agent).sample(mask)
        else:
            allowed_indices = possible_indices[mask == 1]
            max_val = np.finfo(float).min
            next_position_index = 0
            for idx in range(len(allowed_indices)):
                next_board = np.array(_env.board.squares)
                next_board[allowed_indices[idx]] = 1 if agent == 'player_1' else 2
                next_board_hash = str(next_board)
                value = 0 if _q_table[agent].get(next_board_hash) is None else _q_table[agent].get(next_board_hash)
                if value > max_val:
                    max_val = value
                    next_position_index = idx
            action = allowed_indices[next_position_index]

        _env.step(action)
        if training:
            state_table[agent].append(np.array(_env.board.squares))
        observation, reward, termination, truncation, info = _env.last()

        if not training:
            cache_board(cache, rounds, _env.board)
        if termination:
            _winner, indexes = check_winner(cache[rounds], _env.board.winning_combinations)
            if _winner:
                winner_name = agent
                if training:
                    give_reward(_q_table[agent], state_table[agent], 1.0)
                    give_reward(_q_table[other_agent], state_table[other_agent], 0.0)
                elif show:
                    print(f"{COLOR_WIN}won by: {agent}{COLOR_DEFAULT}")
            else:
                winner_name = "draw"
                if training:
                    give_reward(_q_table[agent], state_table[agent], 0.0)
                    give_reward(_q_table[other_agent], state_table[other_agent], 0.0)
                elif show:
                    print("draw")

        rounds += 1

    if not training and show:
        print_cache(cache, rounds, indexes)
    return winner_name


def give_reward(_q_table, states, reward):
    for state in reversed(states):
        state_hash = str(state)
        if _q_table.get(state_hash) is None:
            _q_table[state_hash] = 0.0
        _q_table[state_hash] += learning_rate * (gamma_decay * reward - _q_table[state_hash])
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
    q_table = {env.agents[0]: {}, env.agents[1]: {}}
    if os.path.isfile(file_name):
        os.remove(file_name)
    with open(file_name, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["#of-trainings\tp1-training-won\t#p2-training-won\tdraw-training\t#of-test-games\tp1-test-won\tp2-test-won\tdraw-test\tp1-q-table-size\tp2-q-table-size"])
    number_of_round = 0
    for i in [10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 2000, 4000, 6000, 8000, 10000, 20000, 30000, 40000,
                  50000, 60000, 70000, 80000, 90000, 100000]:
        number_of_round += 1
        games_training = {"draw": 0, "player_1": 0, "player_2": 0}
        print("RL training start ***")
        print(f"iteration: {i}")
        number_of_iteration = 0
        for j in range(1, i+1):
            number_of_iteration += 1
            winner = run(env, q_table, 0.3, True, False)
            games_training[winner] += 1
            env.reset()
        print_summary(games_training)
        print("RL training end ***")
        print("RL test start ***")
        games_test = {"draw": 0, "player_1": 0, "player_2": 0}
        number_of_iteration = 0
        for j in range(1000):
            number_of_iteration += 1
            winner = run(env, q_table, 0.0, False, False)
            games_test[winner] += 1
            env.reset()
        print_summary(games_test)
        with open(file_name, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([f"{i}\t{games_training['player_1']}\t{games_training['player_2']}\t{games_training['draw']}\t1000\t{games_test['player_1']}\t{games_test['player_2']}\t{games_test['draw']}\t{len(q_table['player_1'])}\t{len(q_table['player_2'])}"])
        print("RL test end ***")
    env.close()
    end = time.time()
    print(f"duration: {(end - start):0.1f}ms")
