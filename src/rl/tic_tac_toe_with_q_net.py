# inspired by https://nestedsoftware.com/2019/12/27/tic-tac-toe-with-a-neural-network-1fjn.206436.html
# and https://medium.com/towards-data-science/hands-on-deep-q-learning-9073040ce841

'''
Für Tic-Tac-Toe gibt es 255.168 verschiedene Spielverläufe, von denen 131.184 mit einem Sieg des
ersten Spielers enden, 77.904 mit einem Sieg des zweiten Spielers und 46.080 mit einem Unentschieden
'''
import os
import csv
from pettingzoo.classic import tictactoe_v3
import src.train
from src.rl.tic_tac_toe_utilities import *
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple


EPS_DECAY = 0.95 # eps gets multiplied by this number each epoch...
MIN_EPS = 0.1 # ...until this minimum eps is reached
GAMMA = 0.95 # discount
MAX_MEMORY_SIZE = 10000 # size of the replay memory
BATCH_SIZE = 50 # batch size of the neural network training
MIN_LENGTH = 50 # minimum length of the replay memory for training, before it reached this length, no gradient updates happen

POSSIBLE_INDEXES = np.arange(9)
BOARD_SIZE = 9

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()

        self.layer1 = nn.Linear(BOARD_SIZE, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, 128)
        self.layer5 = nn.Linear(128, BOARD_SIZE)
        self.double()

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)


class QNetContext:
    def __init__(self):
        self.policy_net = QNet().to(device)
        self.policy_net = self.policy_net.train()
        self.target_net = QNet().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net = self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.loss_function = nn.MSELoss()

    def optimize(self, replay_memory, batch_size):
        batch = replay_memory.structured_sample(batch_size)  # get samples from the replay memory

        self.optimizer.zero_grad()
        with torch.no_grad():
            expected_indexes = batch["reward"] + GAMMA * self.target_net(batch["next_state"]).argmax(axis=1) * (1 - batch["done"])  # R(s, a) + γ·maxₐ N(s') if not a terminal state, otherwise R(s, a)
        predicted = self.policy_net(batch["state"])

        for i in range(batch_size): # set the target for the action that was done and leave the outputs of other actions as they are
            predicted[i][batch["action"][i]] = expected_indexes[i]

        loss = self.loss_function(batch["state"], predicted)

        loss.backward()
        self.optimizer.step()

        return loss.item()


def board_to_tensor(state):
    return torch.tensor(np.array([state.copy()]), device=device, dtype=float).flatten()


device = src.train.get_device()
q_net = QNetContext()
env = tictactoe_v3.raw_env()
env.reset(seed=42)
eps = 0.3 # exploration rate, probability of choosing random action
memory_parts = ["state_history", "reward", "done"]
Memory = namedtuple("Memory", memory_parts)
state_parts = ["state_from", "state_to", "action"]
StateMemory = namedtuple("StateMemory", state_parts)


def get_reward(agent):
    winner, _ = check_winner(env.board.squares, env.board.winning_combinations)
    if winner and agent == "player_1":
        return 1
    elif winner and agent == "player_2":
        return -1
    else:
        return 0


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
        batch = self._sample(k)
        intermediate = {"state":[],"action":[],"next_state":[],"reward":[],"done":[]}
        for b in batch:
            state_history = b[0]
            reward = b[1] * 1.33
            done = b[2]
            for s in state_history:
                reward *= 0.75
                intermediate["done"].append(done)
                intermediate["reward"].append(reward)
                intermediate["state"].append(s[0])
                intermediate["next_state"].append(s[1])
                intermediate["action"].append(s[2])
        result = {}
        for key in intermediate.keys():
            if key == "action":
                result[key] = torch.tensor(np.array(intermediate[key]), device=device, dtype=int)
            else:
                result[key] = torch.tensor(np.array(intermediate[key]), device=device, dtype=float)

        return result

    def __len__(self):
        return len(self.memory)


def select_training_action(mask, agent):
    if random.random() < eps or agent == "player_2":
        return env.action_space(agent).sample(mask).item()
    else:
        if sum(mask) == 1:
            return mask.argmax().item()
        output = q_net.policy_net(board_to_tensor(env.board.squares))
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

    file_name = "tic-tac-toe.pth"
    if not os.path.isfile(file_name):

        print("RL training start ***")

        replay_memory = ReplayMemory(max_length=MAX_MEMORY_SIZE)

        best_loss = float('inf')

        for episode in range(10000):

            env.reset()
            rounds = 0
            state_table = deque()

            observation, reward, termination, truncation, info = env.last()

            while not termination:
                agent = env.agents[rounds % 2]
                mask = observation["action_mask"]
                action = select_training_action(mask, agent)

                state = env.board.squares.copy()
                env.step(action)
                observation, reward, termination, truncation, info = env.last()
                if agent == "player_1":
                    state_table.appendleft(StateMemory(state, env.board.squares.copy(), action))
                if termination:
                    reward = get_reward(agent)
                    memory = Memory(state_table.copy(), reward, termination)
                    replay_memory.store(memory)

                if len(replay_memory) >= MIN_LENGTH and agent == "player_1":
                    loss = q_net.optimize(replay_memory, BATCH_SIZE)
                    if loss < best_loss:
                        best_loss = loss
                        q_net.target_net.load_state_dict(q_net.policy_net.state_dict())

                rounds += 1

            eps = max(MIN_EPS, eps * EPS_DECAY)

            if episode % 200 == 0:
                print(f"loss after {episode} episodes: {best_loss:.3}")

        torch.save(q_net.target_net.state_dict(), file_name)

        print(f"RL training end ***")

    print("RL test start ***")

    model = QNet().to(device)
    model.load_state_dict(torch.load(file_name, map_location=device))
    model.eval()

    games = {"draw": 0, "player_1": 0, "player_2": 0}

    for i in range(1000):

        cache = np.zeros((9, 9))
        env.reset()
        rounds = 0
        indexes = [0, 0, 0]
        winner_name = ""

        observation, reward, termination, truncation, info = env.last()

        while not termination:
            agent = env.agents[rounds % 2]
            mask = observation["action_mask"]
            action = select_testing_action(mask, agent, model)

            env.step(action)
            observation, reward, termination, truncation, info = env.last()

            cache_board(cache, rounds, env.board)

            rounds += 1

        winner, indexes = check_winner(cache[rounds-1], env.board.winning_combinations)
        if winner:
            winner_name = agent
            #print(f"{COLOR_WIN}won by: {agent}{COLOR_DEFAULT}")
        else:
            winner_name = "draw"
            #print("draw")
        #print_cache(cache, rounds, indexes)
        games[winner_name] += 1

    print_summary(games)

    print("RL test start ***")
    env.close()
    end = time.time()
    print(f"duration: {(end - start):0.1f}s")
