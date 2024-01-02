# inspired by https://nestedsoftware.com/2019/12/27/tic-tac-toe-with-a-neural-network-1fjn.206436.html
# and https://medium.com/towards-data-science/hands-on-deep-q-learning-9073040ce841

'''
Für Tic-Tac-Toe gibt es 255.168 verschiedene Spielverläufe, von denen 131.184 mit einem Sieg des
ersten Spielers enden, 77.904 mit einem Sieg des zweiten Spielers und 46.080 mit einem Unentschieden
'''
import copy
import os
from pettingzoo.classic import tictactoe_v3
from src.rl.tic_tac_toe_utilities import *
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple


EPS_DECAY = 0.95 # eps gets multiplied by this number each epoch...
MIN_EPS = 0.1 # ...until this minimum eps is reached
GAMMA = 0.95 # discount
MAX_MEMORY_SIZE = 10000 # size of the replay memory
BATCH_SIZE = 32 # batch size of the neural network training

POSSIBLE_INDEXES = np.arange(9)
BOARD_SIZE = 9
IN_FEATURES = BOARD_SIZE
OUT_FEATURES = 16


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(IN_FEATURES, OUT_FEATURES),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(OUT_FEATURES, OUT_FEATURES),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(OUT_FEATURES, BOARD_SIZE),
            nn.Sigmoid()
        )
        self.double()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.output_layer(x)


class QNetContext:
    def __init__(self):
        self.policy_net = QNet().to(device)
        self.policy_net = self.policy_net
        self.target_net = QNet().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net = self.target_net.eval()

        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=1e-4)
        self.loss_function = nn.MSELoss()

    def optimize(self, replay_memory, batch_size):
        batch = replay_memory.structured_sample(batch_size) # get samples from the replay memory
        sum_loss = 0.0
        for b in batch:
            state_history = b[0]
            reward = b[1]
            state_next, action_next = state_history[0]
            sum_loss += self.backpropagate(state_next, action_next, reward)

            for state, action_next in list(state_history)[1:]:
                with torch.no_grad():
                    next_q_values = self.get_q_values(state_next, self.target_net)
                    q_value_max = torch.max(next_q_values).item()
                sum_loss += self.backpropagate(state, action_next, q_value_max * GAMMA)
                state_next = state

        return sum_loss

    def backpropagate(self, state, action, reward):
        self.optimizer.zero_grad()
        output = self.policy_net(board_to_tensor(state))

        target = output.clone().detach()
        target[action] = reward
        illegal_actions = [i for i in range(len(state)) if state[i] != 0]
        for a in illegal_actions:
            target[a] = 0.0

        loss = self.loss_function(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_q_values(self, state, model):
        inputs = board_to_tensor(state)
        outputs = model(inputs)
        return outputs


def board_to_tensor(state):
    return torch.tensor(np.array([state.copy()]), device=device, dtype=float).flatten()


device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
q_net = QNetContext()
env = tictactoe_v3.raw_env()
env.reset(seed=42)
eps = 0.3 # exploration rate, probability of choosing random action
memory_parts = ["state_history", "reward"]
Memory = namedtuple("Memory", memory_parts)


def get_winner_name(_winner, player):
    if _winner:
        return player
    else:
        return "draw"


class ReplayMemory:
    def __init__(self, max_length=None):
        self.max_length = max_length
        self.memory = deque(maxlen=max_length)

    def store(self, data):
        self.memory.append(data)

    def _sample(self, k):
        return random.sample(self.memory, k)

    def structured_sample(self, k):
        return self._sample(k)


    def __len__(self):
        return len(self.memory)


def select_training_action(mask, agent):
    if random.random() < eps or agent == "player_2":
        return env.action_space(agent).sample(mask).item()
    else:
        if sum(mask) == 1:
            return mask.argmax().item()
        output = q_net.get_q_values(env.board.squares, q_net.policy_net)
        output[mask == 0] = min(torch.min(output), 0)
        return output.argmax().item()


def select_testing_action(mask, agent, model):
    if agent == "player_2":
        return env.action_space(agent).sample(mask)
    else:
        if sum(mask) == 1:
            return mask.argmax()
        output = model(board_to_tensor(env.board.squares))
        output[mask == 0] = min(torch.min(output), 0)
        return output.argmax().item()


if __name__ == "__main__":

    start = time.time()

    file_name = "statistics/tic-tac-toe-statistics-q-net.csv"
    if os.path.isfile(file_name):
        os.remove(file_name)
    with open(file_name, 'a') as f:
        f.write("#of-trainings\tp1-training-won\tp2-training-won\tdraw-training\t#of-test-games\tp1-test-won\tp2-test-won\tdraw-test\tbest-loss\n")

    games_training = {"draw": 0, "player_1": 0, "player_2": 0}

    print("RL training start ***")

    replay_memory = ReplayMemory(max_length=MAX_MEMORY_SIZE)

    best_loss = float('inf')

    for episode_training in range(1, 100001):

        env.reset()
        rounds = 0
        state_table = deque()

        observation, reward, termination, truncation, info = env.last()

        winner_name = ""

        while not termination:
            agent = env.agents[rounds % 2]
            mask = observation["action_mask"]
            action = select_training_action(mask, agent)

            if agent == "player_1":
                state_table.appendleft((env.board.squares.copy(), action))

            env.step(action)
            observation, reward, termination, truncation, info = env.last()

            rounds += 1

        reward, winner_name = get_reward(env, agent)
        memory = Memory(state_table.copy(), reward)
        replay_memory.store(memory)

        if len(replay_memory) >= BATCH_SIZE:
            loss = q_net.optimize(replay_memory, BATCH_SIZE)
            if loss < best_loss:
                best_loss = loss
                q_net.target_net.load_state_dict(q_net.policy_net.state_dict())

        eps = max(MIN_EPS, eps * EPS_DECAY)
        games_training[winner_name] += 1

        if episode_training % 100 == 0 and episode_training != 0:
            print(f"episode: {episode_training} - best-loss: {best_loss}")

        if episode_training % 1000 == 0 and episode_training != 0:

            print("-----------------------------")
            print(f"Tests episode: {episode_training}")

            games_test = {"draw": 0, "player_1": 0, "player_2": 0}

            model = copy.deepcopy(q_net.target_net)
            model.eval()

            for episode_test in range(1000):

                env.reset()
                cache = np.zeros((9, 9))
                winner_name = ""
                rounds_test = 0

                observation, reward, termination, truncation, info = env.last()

                while not termination:
                    agent = env.agents[rounds_test % 2]
                    mask = observation["action_mask"]
                    action = select_testing_action(mask, agent, model)

                    env.step(action)
                    observation, reward, termination, truncation, info = env.last()
                    cache_board(cache, rounds_test, env.board)

                    rounds_test += 1

                winner, _ = check_winner(cache[rounds_test-1], env.board.winning_combinations)
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
                f.write(f"{episode_training}\t{games_training['player_1']}\t{games_training['player_2']}\t{games_training['draw']}\t1000\t{games_test['player_1']}\t{games_test['player_2']}\t{games_test['draw']}\t{best_loss}\n")

    torch.save(q_net.target_net.state_dict(), "model/tic-tac-toe.pth")

    print("-----------------------------")
    print("RL training end ***")

    env.close()
    end = time.time()
    with open(file_name, 'a') as f:
        f.write(f"duration: {(end - start):0.1f}s\n")
    print(f"duration: {(end - start):0.1f}s")
