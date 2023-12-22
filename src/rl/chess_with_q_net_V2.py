# inspired by: https://medium.com/towards-data-science/hands-on-deep-q-learning-9073040ce841
# elo vs depth: http://web.ist.utl.pt/diogo.ferreira/papers/ferreira13impact.pdf
# elo vs time-limit: https://www.chess.com/forum/view/general/i-want-a-good-analysis-with-stockfish-how-long-should-i-do-1-min-1-hour-1-day#:~:text=1%20min%20per%20position%20of%20analysis%20will%20be,rating%20may%20be%20as%20low%20as%20900%20rating.

import math
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import chess.engine
from collections import deque
import src.rl.chess_with_q_net_utilities as utils
import src.lib.position_validator as pv

BATCH_SIZE = 8
MAX_MEMORY_SIZE = 1000


class QNetContext:
    def __init__(self, device):
        self.device = device
        self.policy_net = utils.QNet().to(device)
        self.policy_net = self.policy_net.train()
        self.target_net = utils.QNet().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net = self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=utils.LR, amsgrad=True)
        self.loss_function = nn.HuberLoss()
        self.board = chess.Board()

    def optimize(self, replay_memory):
        if len(replay_memory) < BATCH_SIZE:
            return

        mini_batch = replay_memory.structured_sample(BATCH_SIZE) # get samples from the replay memory

        sum_loss = 0.0
        counter = 0
        print("batch:", end='')
        for batch in mini_batch:
            counter += 1
            print(f' {counter}', end='')
            state_history = batch[0]
            reward = batch[1]
            state_next_fen, action_next = state_history[0]
            sum_loss += self.backpropagate(state_next_fen, action_next, reward)

            for state_fen, action_next in list(state_history)[1:]:
                with torch.no_grad():
                    self.board.set_fen(state_next_fen)
                    next_q_values = utils.get_q_values(self.board, self.target_net)
                    q_value_max = torch.max(next_q_values).item()
                sum_loss += self.backpropagate(state_fen, action_next, q_value_max * utils.GAMMA)
                state_next_fen = state_fen

        print()
        return sum_loss

    def backpropagate(self, state_fen, action, reward):
        self.board.set_fen(state_fen)
        state_before = utils.board_to_array(self.board)

        self.optimizer.zero_grad()
        output = self.policy_net(torch.tensor(state_before, device=self.device, dtype=torch.double))

        target = output.clone().detach()
        target[action] = reward

        valid_positions = pv.get_valid_positions(state_fen)
        legal_actions = np.zeros(utils.BOARD_SIZE)
        for position in valid_positions:
            self.board.set_fen(position)
            state_after = utils.board_to_array(self.board)
            _, action_to, _, _ = utils.evaluate_moves(state_before, state_after)
            legal_actions[action_to] = 1.0

        illegal_actions = [i for i in range(len(legal_actions)) if legal_actions[i] == 0]
        for a in illegal_actions:
            target[a] = 0.0

        loss = self.loss_function(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

MAX_ROUNDS = 50
EPISODES_TEST = 100
EPISODES_TRAIN = 1000

if __name__ == "__main__":

    reasons = []
    q_net = QNetContext(utils.device)
    env = chess.engine.SimpleEngine.popen_uci(utils.PATH_TO_CHESS_ENGINE)
    board = chess.Board()

    eps_threshold = utils.EPS_START

    start_overall = time.time()

    best_loss = float('inf')

    file_name = os.path.join("src", "rl", "chess-statistics-q-net-V2.csv")
    if os.path.isfile(file_name):
        os.remove(file_name)
    with open(file_name, 'a') as f:
        f.write("#of-trainings\tp1-training-won\tp2-training-won\tdraw-training\t#of-test-games\tp1-test-won\tp2-test-won\tdraw-test\tbest-loss\taverage-rounds-played\tduration\n")

    games_training = {"draw": 0, "white": 0, "black": 0}

    print("RL training start ***")

    replay_memory = utils.ReplayMemory(max_length=MAX_MEMORY_SIZE)

    # Keep track of steps since model and target model were updated
    steps_since_model_optimize = 0
    steps_since_model_update = 0

    start_episode = time.time()

    rounds_training_total = 0

    for episode_training in range(1, EPISODES_TRAIN+1):

        board.set_fen("3k4/8/3K4/3P4/8/8/8/8 w - - 0 1")
        rounds_training = 0
        winner_name = "draw"
        state_table = deque()

        while not board.is_game_over() and rounds_training < MAX_ROUNDS:
            action_from_white, action_to_white, promotion, reward_after_white_move = utils.select_action(env, board, eps_threshold,True, q_net.policy_net, True)
            action_from_str, action_to_str = utils.get_actions(action_from_white, action_to_white, promotion)
            state_table.appendleft([board.fen(), action_to_white])
            utils.play_and_print(board, rounds_training, "white", action_from_str, action_to_str)

            if board.is_game_over():
                replay_memory.store([state_table, reward_after_white_move])
                reason = utils.board_status.reason_why_the_game_is_over(board)
                if reason not in reasons:
                    reasons.append(reason)
                if utils.white_is_winner(reason):
                    winner_name = "white"
            else:
                action_from_black, action_to_black, promotion, reward_after_black_move = utils.select_action(env, board, eps_threshold,False, q_net.policy_net, True)
                action_from_str, action_to_str = utils.get_actions(action_from_black, action_to_black, promotion)
                utils.play_and_print(board, rounds_training, "black", action_from_str, action_to_str)
                if board.is_game_over():
                    replay_memory.store([state_table, reward_after_black_move])
                    reason = utils.board_status.reason_why_the_game_is_over(board)
                    if reason not in reasons:
                        reasons.append(reason)
                    if utils.black_is_winner(reason):
                        winner_name = "black"

            rounds_training += 1
            steps_since_model_optimize += 1
            steps_since_model_update += 1

        rounds_training_total += rounds_training

        if steps_since_model_optimize >= 5:
            loss = q_net.optimize(replay_memory)
            if loss is not None and loss < best_loss:
                best_loss = loss
            steps_since_model_optimize = 0

        if steps_since_model_update >= 100:
            q_net.target_net.load_state_dict(q_net.policy_net.state_dict())
            steps_since_model_update = 0

        eps_threshold = utils.EPS_END + (utils.EPS_START - utils.EPS_END) * math.exp(-1. * rounds_training_total / utils.EPS_DECAY)
        games_training[winner_name] += 1

        if episode_training % (EPISODES_TEST/10) == 0 and episode_training != 0:
            print(f"episode: {episode_training:6} - best-loss: {best_loss:0.16f} - average rounds played: {(rounds_training_total/EPISODES_TEST*10):0.1f}")
            rounds_training_total = 0

        if episode_training % EPISODES_TEST == 0 and episode_training != 0:

            end = time.time()
            print(f"duration: {(end - start_episode):0.1f}s")

            start_training = time.time()

            print("-----------------------------")
            print(f"Tests episode: {episode_training}")

            games_test = {"draw": 0, "white": 0, "black": 0}

            model = copy.deepcopy(q_net.target_net)
            model.eval()

            rounds_test_total = 0

            for episode_test in range(EPISODES_TEST):

                board.set_fen("3k4/8/3K4/3P4/8/8/8/8 w - - 0 1")
                rounds_test = 0
                winner_name = "draw"

                while not board.is_game_over() and rounds_test < MAX_ROUNDS:
                    action_from_white, action_to_white, promotion, reward_after_white_move = utils.select_action(env, board, eps_threshold,True, model, False)
                    action_from_str, action_to_str = utils.get_actions(action_from_white, action_to_white, promotion)
                    utils.play(board, action_from_str, action_to_str)

                    if board.is_game_over():
                        reason = utils.board_status.reason_why_the_game_is_over(board)
                        if reason not in reasons:
                            reasons.append(reason)
                        if utils.white_is_winner(reason):
                            winner_name = "white"
                    else:
                        action_from_white, action_to_white, promotion, reward_after_white_move = utils.select_action(env, board, eps_threshold,False, model, False)
                        action_from_str, action_to_str = utils.get_actions(action_from_white, action_to_white, promotion)
                        utils.play(board, action_from_str, action_to_str)
                        if board.is_game_over():
                            reason = utils.board_status.reason_why_the_game_is_over(board)
                            if reason not in reasons:
                                reasons.append(reason)
                            if utils.black_is_winner(reason):
                                winner_name = "black"

                    rounds_test += 1

                games_test[winner_name] += 1
                rounds_test_total += rounds_test

            print("Player     Training     Test")
            print(f"draw:      {games_training['draw']:8}\t{games_test['draw']:8}")
            print(f"white:     {games_training['white']:8}\t{games_test['white']:8}")
            print(f"black:     {games_training['black']:8}\t{games_test['black']:8}")
            print(f"average rounds played: {(rounds_test_total/EPISODES_TEST):0.1f}")

            end = time.time()
            print(f"duration: {(end - start_training):0.1f}s")

            with open(file_name, 'a') as f:
                f.write(f"{episode_training}\t{games_training['white']}\t{games_training['black']}\t{games_training['draw']}\t{EPISODES_TEST}\t{games_test['white']}\t{games_test['black']}\t{games_test['draw']}\t{best_loss}\t{(rounds_test_total/EPISODES_TEST):0.1f}\t{(end - start_training):0.1f}\n")

            start_episode = time.time()

    torch.save(q_net.target_net.state_dict(), os.path.join("src", "rl", "chess.pth"))

    print(reasons)

    print("-----------------------------")
    print("RL training end ***")

    env.close()
    end = time.time()
    with open(file_name, 'a') as f:
        f.write(f"duration: {(end - start_overall):0.1f}s\n")
    print(f"duration: {(end - start_overall):0.1f}s")
