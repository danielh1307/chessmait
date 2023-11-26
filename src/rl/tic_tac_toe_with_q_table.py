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
    file_name = "tic-tac-toe-statistics.csv"
    print("RL training start ***")
    env = tictactoe_v3.raw_env()
    env.reset(seed=42)
    q_table = {env.agents[0]: {}, env.agents[1]: {}}
    if os.path.isfile(file_name):
        os.remove(file_name)
    number_of_round = 0
    for i in range(1, 1001):
        number_of_round += 1
        games = {}
        print(f"iteration: {i}")
        number_of_iteration = 0
        for j in range(1, (i*100)+1):
            number_of_iteration += 1
            winner = run(env, q_table, 0.3, True, False)
            if winner not in games.keys():
                games[winner] = 0
            games[winner] += 1
            env.reset()
        print_summary(games)
        with open(file_name, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([f"--- training round {number_of_round} with {number_of_iteration} # of iterations ------------"])
            csvwriter.writerow(["--- player\t#-of-games\tq-table-size"])
            for k in sorted(games.keys()):
                if k == 'draw':
                    csvwriter.writerow([f"{k}\t{games[k]}\t"])
                else:
                    csvwriter.writerow([f"{k}\t{games[k]}\t{len(q_table[k])}"])
        print(f"q-table size: {get_size_of_q_table(q_table)}")
        print(f"q-table entries player-1: {len(q_table['player_1'])}")
        print(f"q-table entries player-2: {len(q_table['player_2'])}")
        print("RL training end ***")
        print("RL gaming start ***")
        games = {}
        number_of_iteration = 0
        for j in range(1000):
            number_of_iteration += 1
            winner = run(env, q_table, 0.0, False, False)
            if winner not in games.keys():
                games[winner] = 0
            games[winner] += 1
            env.reset()
        print_summary(games)
        with open(file_name, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([f"*** gaming with {number_of_iteration} # of iterations"])
            csvwriter.writerow(["*** player\t#-of-games"])
            for k in sorted(games.keys()):
                csvwriter.writerow([f"{k}\t{games[k]}"])
    print("RL gaming end ***")
    env.close()
