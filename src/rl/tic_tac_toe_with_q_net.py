# inspired by https://nestedsoftware.com/2019/12/27/tic-tac-toe-with-a-neural-network-1fjn.206436.html

'''
Für Tic-Tac-Toe gibt es 255.168 verschiedene Spielverläufe, von denen 131.184 mit einem Sieg des
ersten Spielers enden, 77.904 mit einem Sieg des zweiten Spielers und 46.080 mit einem Unentschieden
'''
import os
import sys

import numpy as np
from pettingzoo.classic import tictactoe_v3

import src.train
from src.rl.tic_tac_toe_utilities import *
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.3
LR = 0.1

POSSIBLE_INDEXES = np.arange(9)
MAX_NUMBER_OF_ITERATIONS = 101

BOARD_SIZE = 9
NUM_OF_BOARD_REPRESENTATIONS = 3 # one for player-1 fields, one for player-2 fields, one for empty fields


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()

        self.layer1 = nn.Linear(BOARD_SIZE * NUM_OF_BOARD_REPRESENTATIONS, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, BOARD_SIZE)
        self.double()

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class QNetContext:
    def __init__(self):
        self.policy_net = QNet().to(device)
        self.policy_net = self.policy_net.train()
        self.target_net = QNet().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net = self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.loss_function = nn.MSELoss()

    def optimize(self, states, reward):
        last_state = states[-1]
        self.back_propagate(last_state, reward)
        del states[-1]
        for state in reversed(states):
            with torch.no_grad():
                next_best = self.target_net(board_to_tensor(last_state[0])).argmax().item()
            self.back_propagate(state, reward * next_best)
            last_state = state
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def back_propagate(self, state, reward):
        self.optimizer.zero_grad()
        predicted = self.policy_net(board_to_tensor(state[0]))
        # set up expected
        expected = predicted.clone().detach()
        expected[state[1]] = reward
        expected[state[0] == 0] = 0 # set loss of not set fields to zero
        loss = self.loss_function(predicted, expected)
        loss.backward()
        self.optimizer.step()


def board_to_tensor(state):
    result = np.array([state.copy(),state.copy(),state.copy()])
    result[0][result[0] != 1] = 0
    result[1][result[1] != 2] = 0
    result[1][result[1] == 2] = 1
    result[2][result[2] == 0] = 3
    result[2][result[2] != 3] = 0
    result[2][result[2] == 3] = 1
    return torch.tensor(result, device=device, dtype=float).flatten()


def run(_env, training, num_of_iteration):

    observation, reward, termination, truncation, info = _env.last()

    rounds = 0
    state_table = []

    while not termination and rounds < 10:

        agent = _env.agents[rounds % 2]

        mask = observation["action_mask"]
        action = select_action(_env, state_table, mask, agent, training, num_of_iteration)

        _env.step(action.item())
        observation, reward, termination, truncation, info = _env.last()
        state_table.append((np.array(_env.board.squares), action))

        if termination:
            _winner, _ = check_winner(state_table[rounds][0], _env.board.winning_combinations)
            q_net.optimize(state_table, get_reward(_winner, agent))

        rounds += 1

    return get_winner_name(_winner, agent)


def get_reward(_winner, player):
    if _winner and player == "player_1":
        return 1
    elif _winner and player == "player_2":
        return -1
    else:
        return 0


def get_winner_name(_winner, player):
    if _winner:
        return player
    else:
        return "draw"


def select_action(_env, states, _mask, _agent, _training, num_of_iteration):
    sample = np.random.random()
    eps_threshold = EPS_START - (EPS_START - EPS_END) * num_of_iteration / MAX_NUMBER_OF_ITERATIONS
    if _agent == "player_1" and eps_threshold > sample and _training or _agent == "player_2":
        return torch.tensor([_env.action_space(_agent).sample(_mask)], device=device, dtype=torch.long)
    else:
        if sum(_mask) == 1:
            return _mask.argmax()
        if not states:
            state = np.zeros(BOARD_SIZE)
        else:
            state = states[-1][0]
        output = q_net.target_net(board_to_tensor(state))
        output[_mask == 0] = min(torch.min(output), 0)
        return output.argmax()


device = src.train.get_device()
q_net = QNetContext()


if __name__ == "__main__":
    start = time.time()
    env = tictactoe_v3.raw_env()
    env.reset(seed=42)

    games_training = {"draw": 0, "player_1": 0, "player_2": 0}
    print("RL training start ***")
    number_of_iterations = 0
    for i in range(MAX_NUMBER_OF_ITERATIONS):
        number_of_iterations += 1
        winner = run(env, True, i)
        games_training[winner] += 1
        env.reset()
        if (number_of_iterations-1) in [10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 2000, 4000, 6000, 8000,
                                        10_000, 20_000, 40_000, 60_000, 80_000, 100_000, 200_000, 400_000, 600_000,
                                        800_000, 1_000_000]:
            print(f"iteration: {i}")
            print_summary(games_training)
            print("RL test start ***")
            games_test = {"draw": 0, "player_1": 0, "player_2": 0}
            for j in range(100):
                winner = run(env, False, i)
                games_test[winner] += 1
                env.reset()
            print_summary(games_test)
            print("RL test end ***")
    print("RL training end ***")
    env.close()
    end = time.time()
    print(f"duration: {(end - start):0.1f}s")
