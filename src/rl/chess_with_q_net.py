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


OPTIMIZER = 4
LOSS_FUNCTION = 2


class QNetContext:
    def __init__(self, device):
        self.device = device
        self.policy_net = utils.QNet().to(device)
        self.policy_net = self.policy_net.train()
        self.target_net = utils.QNet().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net = self.target_net.eval()

        self.optimizer1 = optim.SGD(self.policy_net.parameters(), lr=utils.LR)
        self.optimizer2 = optim.Adagrad(self.policy_net.parameters(), lr=utils.LR)
        self.optimizer3 = optim.Adam(self.policy_net.parameters(), lr=utils.LR)
        self.optimizer4 = optim.AdamW(self.policy_net.parameters(), lr=utils.LR, amsgrad=True)

        self.loss_function1 = nn.MSELoss()
        self.loss_function2 = nn.HuberLoss()
        self.loss_function3 = nn.SmoothL1Loss()

    def optimize(self, replay_memory):
        if len(replay_memory) < utils.BATCH_SIZE:
            return

        mini_batch = replay_memory.structured_sample(utils.BATCH_SIZE) # get samples from the replay memory

        # Get the Q values for the initial states of the trajectories from the model
        initial_states = np.array([batch[0] for batch in mini_batch])
        initial_qs = self.policy_net(torch.tensor(initial_states, device=self.device, dtype=torch.double))

        # Get the "target" Q values for the next states
        next_states = np.array([batch[3] for batch in mini_batch])
        target_qs = self.target_net(torch.tensor(next_states, device=self.device, dtype=torch.double))

        states = np.empty([len(mini_batch), utils.BOARD_SIZE])
        updated_qs = torch.empty((len(mini_batch), utils.BOARD_SIZE), device=self.device, dtype=torch.double)

        for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
            if not done:
                # If not terminal, include the next state
                max_future_q = reward + utils.GAMMA * torch.max(target_qs[index]).item()
            else:
                # If terminal, only include the immediate reward
                max_future_q = reward

            # The Qs for this sample of the mini batch
            updated_qs_sample = initial_qs[index].clone().detach()
            # Update the value for the taken action
            action_to = action[0].item()
            updated_qs_sample[action_to] = max_future_q

            # Keep track of the observation and updated Q value
            states[index] = observation
            updated_qs[index] = updated_qs_sample

        predicted_qs = self.policy_net(torch.tensor(states, device=self.device, dtype=torch.double))
        if LOSS_FUNCTION == 2:
            loss_function = self.loss_function2
        elif LOSS_FUNCTION == 3:
            loss_function = self.loss_function3
        else:
            loss_function = self.loss_function1

        if OPTIMIZER == 2:
            optimizer = self.optimizer2
        elif OPTIMIZER == 3:
            optimizer = self.optimizer3
        elif OPTIMIZER == 4:
            optimizer = self.optimizer4
        else:
            optimizer = self.optimizer1

        loss = loss_function(predicted_qs, updated_qs)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(q_net.policy_net.parameters(), 100)
        optimizer.step()
        return loss

def write_replay_memory(state_table, replay_memory, rounds_training):
    reward = state_table[0][2]
    decrease_reward = 0.01 if reward > 0 else -0.01
    if reward == utils.NO_REWARD:
        reward = -0.01 * rounds_training
    for state in state_table:
        replay_memory.store([state[0], state[1], reward, state[3], state[4]])
        reward -= decrease_reward


EPISODES_TEST = 1000
EPISODES_TRAIN = 100000

if __name__ == "__main__":

    reasons = []
    q_net = QNetContext(utils.device)
    env = chess.engine.SimpleEngine.popen_uci(utils.PATH_TO_CHESS_ENGINE)
    board = chess.Board()

    eps_threshold = utils.EPS_START

    start_overall = time.time()

    best_loss = float('inf')

    file_name = os.path.join("src", "rl", "chess-statistics-q-net.csv")
    if os.path.isfile(file_name):
        os.remove(file_name)
    with open(file_name, 'a') as f:
        f.write("#of-trainings\tp1-training-won\tp2-training-won\tdraw-training\t#of-test-games\tp1-test-won\tp2-test-won\tdraw-test\tbest-loss\taverage-rounds-played\tduration\n")

    games_training = {"draw": 0, "white": 0, "black": 0}

    print("RL training start ***")

    replay_memory = utils.ReplayMemory(max_length=utils.MAX_MEMORY_SIZE)

    # Keep track of steps since model and target model were updated
    steps_since_model_update = 0

    start_episode = time.time()

    rounds_training_total = 0

    for episode_training in range(1, EPISODES_TRAIN+1):

        board.set_fen("3k4/8/3K4/3P4/8/8/8/8 w - - 0 1")
        rounds_training = 0
        winner_name = "draw"
        state_table = deque()

        insufficient_material = False

        while not board.is_checkmate() and not board.is_stalemate() and not board.is_fifty_moves() and not board.is_fivefold_repetition() and not insufficient_material:
            action_from_white, action_to_white, promotion, reward_after_white_move = utils.select_action(env, board, eps_threshold,True, q_net.policy_net, True)
            action_from_str, action_to_str = utils.get_actions(action_from_white, action_to_white, promotion)
            state_table_before = utils.board_to_array(board)
            utils.play_and_print(board, rounds_training, "white", action_from_str, action_to_str)
            state_table_after = utils.board_to_array(board)
            insufficient_material = utils.is_unsufficient_material(state_table_after)

            if reward_after_white_move == utils.WIN_REWARD and not board.is_checkmate() and not board.is_stalemate() and not insufficient_material:
                state_table.appendleft([state_table_before, action_to_white, reward_after_white_move, state_table_after, False])
                write_replay_memory(state_table, replay_memory, rounds_training)
                state_table = deque()
            if board.is_checkmate() or board.is_stalemate() or insufficient_material:
                state_table.appendleft([state_table_before, action_to_white, reward_after_white_move, state_table_after, True])
                reason = utils.board_status.reason_why_the_game_is_over(board)
                if reason not in reasons:
                    reasons.append(reason)
                if utils.white_is_winner(reason):
                    winner_name = "white"
            else:
                action_from_black, action_to_black, promotion, reward_after_black_move = utils.select_action(env, board, eps_threshold,False, q_net.policy_net, True)
                action_from_str, action_to_str = utils.get_actions(action_from_black, action_to_black, promotion)
                utils.play_and_print(board, rounds_training, "black", action_from_str, action_to_str)
                state_table_after = utils.board_to_array(board)
                insufficient_material = utils.is_unsufficient_material(state_table_after)

                if board.is_checkmate() or board.is_stalemate() or insufficient_material:
                    reason = utils.board_status.reason_why_the_game_is_over(board)
                    state_table.appendleft([state_table_before, action_to_white, reward_after_black_move, state_table_after, True])
                    if reason not in reasons:
                        reasons.append(reason)
                    if utils.black_is_winner(reason):
                        winner_name = "black"
                else:
                    if reward_after_white_move != utils.WIN_REWARD:
                        state_table.appendleft([state_table_before, action_to_white, reward_after_white_move, state_table_after, False])

            rounds_training += 1
            steps_since_model_update += 1

        write_replay_memory(state_table, replay_memory, rounds_training)
        rounds_training_total += rounds_training

        if steps_since_model_update >= 10:
            loss = q_net.optimize(replay_memory)
            if loss is not None and loss < best_loss:
                best_loss = loss
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

                insufficient_material = False

                while not board.is_checkmate() and not board.is_stalemate() and not board.is_fifty_moves() and not board.is_fivefold_repetition() and not insufficient_material:
                    action_from_white, action_to_white, promotion, reward_after_white_move = utils.select_action(env, board, eps_threshold,True, model, False)
                    action_from_str, action_to_str = utils.get_actions(action_from_white, action_to_white, promotion)
                    utils.play(board, action_from_str, action_to_str)
                    state_table_after = utils.board_to_array(board)
                    insufficient_material = utils.is_unsufficient_material(state_table_after)

                    if board.is_checkmate() or board.is_stalemate() or insufficient_material:
                        reason = utils.board_status.reason_why_the_game_is_over(board)
                        if reason not in reasons:
                            reasons.append(reason)
                        if utils.white_is_winner(reason):
                            winner_name = "white"
                    else:
                        action_from_white, action_to_white, promotion, reward_after_white_move = utils.select_action(env, board, eps_threshold,False, model, False)
                        action_from_str, action_to_str = utils.get_actions(action_from_white, action_to_white, promotion)
                        utils.play(board, action_from_str, action_to_str)
                        state_table_after = utils.board_to_array(board)
                        insufficient_material = utils.is_unsufficient_material(state_table_after)

                        if board.is_checkmate() or board.is_stalemate() or insufficient_material:
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
