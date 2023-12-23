# inspired by https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542
# and https://nestedsoftware.com/2019/12/27/tic-tac-toe-with-a-neural-network-1fjn.206436.html

'''
Für Tic-Tac-Toe gibt es 255.168 verschiedene Spielverläufe, von denen 131.184 mit einem Sieg des
ersten Spielers enden, 77.904 mit einem Sieg des zweiten Spielers und 46.080 mit einem Unentschieden
'''
import os
import sys
from collections import deque

from pettingzoo.classic import tictactoe_v3
from src.rl.tic_tac_toe_utilities import *
import time

GAMMA = 0.9
EPS_DECAY = 0.95 # eps gets multiplied by this number each epoch...
MIN_EPS = 0.1 # ...until this minimum eps is reached
LR = 0.4

POSSIBLE_INDEXES = np.arange(9)
MAX_NUMBER_OF_ITERATIONS = 1_000_001

env = tictactoe_v3.raw_env()
env.reset(seed=42)
q_table = {}
eps = 0.3


def select_training_action(mask, agent):
    if np.random.random() < eps or agent == "player_2":
        return env.action_space(agent).sample(mask)
    else:
        return get_greedy_action(mask)


def select_testing_action(mask, agent):
    if agent == "player_2":
        return env.action_space(agent).sample(mask)
    else:
        return get_greedy_action(mask)


def get_greedy_action(mask):
    allowed_indices = POSSIBLE_INDEXES[mask == 1]
    max_val = np.finfo(float).min
    next_position_index = 0
    for idx in range(len(allowed_indices)):
        next_board = np.array(env.board.squares)
        next_board[allowed_indices[idx]] = 1
        next_board_hash = str(next_board)
        value = 0 if q_table.get(next_board_hash) is None else q_table.get(next_board_hash)
        if value > max_val:
            max_val = value
            next_position_index = idx
    return allowed_indices[next_position_index]


def give_reward(states, reward):
    state_next, action_next = states[0]
    current_q_value = get_q_value(state_next, action_next)
    new_q_value = calculate_new_q_value(current_q_value, reward,0.0)
    update_q_value(state_next, action_next, new_q_value)

    for state, action_next in list(states)[1:]:

        max_next_q_value = get_max_q_value(state_next)

        current_q_value = get_q_value(state, action_next)
        new_q_value = calculate_new_q_value(current_q_value, 0.0, max_next_q_value)
        update_q_value(state, action_next, new_q_value)

        state_next = state


def get_max_q_value(state):
    actions = [i for i in range(len(state)) if state[i] == 0]
    q_values = [get_q_value(state, action) for action in actions]
    return max(q_values)


def get_q_value(state, action_next):
    state = state.copy()
    state[action_next] = 1
    state_hash = str(state)
    if q_table.get(state_hash) is None:
        q_table[state_hash] = 0.0
    return q_table[state_hash]


def update_q_value(state, action_next, q_value):
    state = state.copy()
    state[action_next] = 1
    state_hash = str(state)
    q_table[state_hash] = q_value


def calculate_new_q_value(current_qvalue, reward, max_next_qvalue):
    return (1 - LR) * current_qvalue + (LR * (reward + GAMMA * max_next_qvalue))


def get_size_of_q_table():
    size_in_in_bytes = sys.getsizeof(q_table)
    for k, v in q_table.items():
        size_in_in_bytes += sys.getsizeof(k)
        size_in_in_bytes += sys.getsizeof(v)
    return size_in_in_bytes


if __name__ == "__main__":

    start = time.time()

    file_name = "statistics/tic-tac-toe-statistics-q-table.csv"
    if os.path.isfile(file_name):
        os.remove(file_name)
    with open(file_name, 'a') as f:
        f.write("#of-trainings\tp1-training-won\tp2-training-won\tdraw-training\t#of-test-games\tp1-test-won\tp2-test-won\tdraw-test\tq-table-size\n")

    games_training = {"draw": 0, "player_1": 0, "player_2": 0}

    print("RL training start ***")

    for episode_training in range(1, 100001):

        env.reset()
        rounds_training = 0
        state_table = deque()
        cache = np.zeros((9, 9))

        observation, reward, termination, truncation, info = env.last()

        winner_name = ""

        while not termination:
            agent = env.agents[rounds_training % 2]
            mask = observation["action_mask"]
            action = select_training_action(mask, agent)

            if agent == "player_1":
                state_table.appendleft((np.array(env.board.squares), action))

            env.step(action)
            observation, reward, termination, truncation, info = env.last()
            cache_board(cache, rounds_training, env.board)

            rounds_training += 1

        eps = max(MIN_EPS, eps * EPS_DECAY)
        winner, indexes = check_winner(cache[rounds_training-1], env.board.winning_combinations)
        if winner:
            winner_name = agent
            if agent == "player_1":
                give_reward(state_table, 1.0)
        else:
            winner_name = "draw"
            if agent == "player_1":
                give_reward(state_table, 0.0)
        games_training[winner_name] += 1

        if episode_training % 1000 == 0 and episode_training != 0:

            print("-----------------------------")
            print(f"Tests episode: {episode_training}")

            games_test = {"draw": 0, "player_1": 0, "player_2": 0}

            for episode_test in range(1000):

                env.reset()
                cache = np.zeros((9, 9))
                winner_name = ""
                rounds_test = 0

                observation, reward, termination, truncation, info = env.last()

                while not termination:
                    agent = env.agents[rounds_test % 2]
                    mask = observation["action_mask"]
                    action = select_testing_action(mask, agent)

                    env.step(action)
                    observation, reward, termination, truncation, info = env.last()
                    cache_board(cache, rounds_test, env.board)

                    rounds_test += 1

                winner, indexes = check_winner(cache[rounds_test-1], env.board.winning_combinations)
                if winner:
                    winner_name = agent
                else:
                    winner_name = "draw"

                games_test[winner_name] += 1

            print("Player     Training     Test")
            print(f"draw:      {games_training['draw']:8}\t{games_test['draw']:8}")
            print(f"player-1:  {games_training['player_1']:8}\t{games_test['player_1']:8}")
            print(f"player-2:  {games_training['player_2']:8}\t{games_test['player_2']:8}")

            with open(file_name, 'a') as f:
                f.write(f"{episode_training}\t{games_training['player_1']}\t{games_training['player_2']}\t{games_training['draw']}\t1000\t{games_test['player_1']}\t{games_test['player_2']}\t{games_test['draw']}\t{len(q_table)}\n")

    print("-----------------------------")
    print("RL training end ***")

    env.close()
    end = time.time()
    with open(file_name, 'a', newline='') as f:
        f.write(f"duration: {(end - start):0.1f}s\n")
    print(f"duration: {(end - start):0.1f}s")
