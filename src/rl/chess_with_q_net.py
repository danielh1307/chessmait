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
import random
import chess.engine
from collections import deque
from src.lib.position_validator import get_valid_positions
from play.board_status import BoardStatus

PATH_TO_CHESS_ENGINE = os.path.join("stockfish", "stockfish-windows-x86-64-avx2.exe")

BATCH_SIZE = 128 # the number of transitions sampled from the replay buffer
GAMMA = 0.99 # the discount factor as mentioned in the previous section
EPS_START = 0.9 # the starting value of epsilon
EPS_END = 0.05 # the final value of epsilon
EPS_DECAY = 1000 # controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # the update rate of the target network
LR = 1e-6 # the learning rate of the optimizer
MAX_MEMORY_SIZE = 10000 # size of the replay memory

POSSIBLE_INDEXES = np.arange(9)
BOARD_SIZE = 64


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(BOARD_SIZE, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(256, BOARD_SIZE),
        )
        self.double()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.output_layer(x)


OPTIMIZER = 3
LOSS_FUNCTION = 3


class QNetContext:
    def __init__(self):
        self.policy_net = QNet().to(device)
        self.policy_net = self.policy_net.train()
        self.target_net = QNet().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net = self.target_net.eval()

        self.optimizer1 = optim.SGD(self.policy_net.parameters(), lr=LR)
        self.optimizer2 = optim.Adagrad(self.policy_net.parameters(), lr=LR)
        self.optimizer3 = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.optimizer4 = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)

        self.loss_function1 = nn.MSELoss()
        self.loss_function2 = nn.HuberLoss()
        self.loss_function3 = nn.SmoothL1Loss()

    def optimize(self, replay_memory):
        if len(replay_memory) < BATCH_SIZE:
            return

        mini_batch = replay_memory.structured_sample(BATCH_SIZE) # get samples from the replay memory

        # Get the Q values for the initial states of the trajectories from the model
        initial_states = np.array([batch[0] for batch in mini_batch])
        initial_qs = self.policy_net(torch.tensor(initial_states, device=device, dtype=torch.double))

        # Get the "target" Q values for the next states
        next_states = np.array([batch[3] for batch in mini_batch])
        target_qs = self.target_net(torch.tensor(next_states, device=device, dtype=torch.double))

        states = np.empty([len(mini_batch), BOARD_SIZE])
        updated_qs = torch.empty((len(mini_batch), BOARD_SIZE), device=device, dtype=torch.double)

        for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
            if not done:
                # If not terminal, include the next state
                max_future_q = reward + GAMMA * torch.max(target_qs[index]).item()
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

        predicted_qs = self.policy_net(torch.tensor(states, device=device, dtype=torch.double))
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

    def get_q_values(self, state, model):
        inputs = board_to_tensor_with_board(state)
        outputs = model(inputs)
        return outputs


def board_to_tensor_with_state(this_state):
    return torch.tensor(this_state, device=device, dtype=float).flatten()


def board_to_tensor_with_board(this_board):
    int_board = board_status.convert_to_int(this_board)
    return torch.tensor(np.array(int_board), device=device, dtype=float).flatten()


def board_to_array(this_board):
    return np.array(board_status.convert_to_int(this_board)).flatten()


device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
q_net = QNetContext()
env = chess.engine.SimpleEngine.popen_uci(PATH_TO_CHESS_ENGINE)
board = chess.Board()
board_status = BoardStatus()
eps_threshold = EPS_START


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


WHITE_PAWN = 1
WHITE_KNIGHT = 2
WHITE_BISHOP = 3
WHITE_ROOK = 4
WHITE_QUEEN = 5
WHITE_KING = 6
BLACK_KING = 20
NO_PROMOTION = 0
NO_REWARD = 0
WIN_REWARD = 1
LOST_REWARD = -1


def evaluate_moves(before, after):
    before[before == -6] = BLACK_KING
    after[after == -6] = BLACK_KING
    after -= before
    if -BLACK_KING in after: # black king moved
        if BLACK_KING in after:
            return np.where(after == -BLACK_KING), np.where(after == BLACK_KING), NO_PROMOTION, NO_REWARD
        if BLACK_KING - WHITE_PAWN in after:
            return np.where(after == -BLACK_KING), np.where(after == BLACK_KING - WHITE_PAWN), NO_PROMOTION, LOST_REWARD # black king has taken the white pawn
        if BLACK_KING - WHITE_KING in after:
            return np.where(after == -BLACK_KING), np.where(after == BLACK_KING - WHITE_KING), NO_PROMOTION, LOST_REWARD # black king has taken the white king
        if BLACK_KING - WHITE_QUEEN in after:
            return np.where(after == -BLACK_KING), np.where(after == BLACK_KING - WHITE_QUEEN), NO_PROMOTION, LOST_REWARD # black king has taken the white queen
        if BLACK_KING - WHITE_ROOK in after:
            return np.where(after == -BLACK_KING), np.where(after == BLACK_KING - WHITE_ROOK), NO_PROMOTION, LOST_REWARD # black king has taken the white rook
        if BLACK_KING - WHITE_BISHOP in after:
            return np.where(after == -BLACK_KING), np.where(after == BLACK_KING - WHITE_BISHOP), NO_PROMOTION, LOST_REWARD # black king has taken the white bishop
        if BLACK_KING - WHITE_KNIGHT in after:
            return np.where(after == -BLACK_KING), np.where(after == BLACK_KING - WHITE_KNIGHT), NO_PROMOTION, LOST_REWARD # black king has taken the white bishop
        else:
            raise Exception("no valid move evaluated for black king")
    elif -WHITE_KING in after:  # white king moved
        if WHITE_KING in after:
            return np.where(after == -WHITE_KING), np.where(after == WHITE_KING), NO_PROMOTION, NO_REWARD
        if BLACK_KING - WHITE_KING in after:
            return np.where(after == -WHITE_KING), np.where(after == BLACK_KING - WHITE_KING), NO_PROMOTION, WIN_REWARD # white king has taken the black king
        else:
            raise Exception("no valid move evaluated for white king")
    elif -WHITE_PAWN in after: # white pawn moved
        if WHITE_PAWN in after:
            return np.where(after == -WHITE_PAWN), np.where(after == WHITE_PAWN), NO_PROMOTION, NO_REWARD
        elif WHITE_KNIGHT in after:
            return np.where(after == -WHITE_PAWN), np.where(after == WHITE_KNIGHT), WHITE_KNIGHT,  WIN_REWARD# convert pawn to knight
        elif WHITE_BISHOP in after:
            return np.where(after == -WHITE_PAWN), np.where(after == WHITE_BISHOP), WHITE_BISHOP, WIN_REWARD # convert pawn to bishop
        elif WHITE_ROOK in after:
            return np.where(after == -WHITE_PAWN), np.where(after == WHITE_ROOK), WHITE_ROOK, WIN_REWARD # convert pawn to rook
        elif WHITE_QUEEN in after:
            return np.where(after == -WHITE_PAWN), np.where(after == WHITE_QUEEN), WHITE_QUEEN, WIN_REWARD # convert pawn to queen
        elif BLACK_KING - WHITE_PAWN in after:
            return np.where(after == -WHITE_PAWN), np.where(after == BLACK_KING - WHITE_PAWN), NO_PROMOTION, WIN_REWARD # white pawn has taken the black king
        else:
            raise Exception("no valid move evaluated for pawn")
    elif -WHITE_QUEEN in after: # white queen moved
        if -WHITE_QUEEN in after:
            return np.where(after == -WHITE_QUEEN), np.where(after == WHITE_QUEEN), NO_PROMOTION, NO_REWARD
        if BLACK_KING - WHITE_QUEEN in after:
            return np.where(after == -WHITE_QUEEN), np.where(after == BLACK_KING - WHITE_QUEEN), NO_PROMOTION, WIN_REWARD # white queen has taken the black king
        else:
            raise Exception("no valid move evaluated for queen")
    elif -WHITE_ROOK in after: # white rook moved
        if -WHITE_ROOK in after:
            return np.where(after == -WHITE_ROOK), np.where(after == WHITE_ROOK), NO_PROMOTION, NO_REWARD
        if BLACK_KING - WHITE_ROOK in after:
            return np.where(after == -WHITE_ROOK), np.where(after == BLACK_KING - WHITE_ROOK), NO_PROMOTION, WIN_REWARD # white rook has taken the black king
        else:
            raise Exception("no valid move evaluated for rook")
    elif -WHITE_BISHOP in after: # white bishop moved
        if -WHITE_BISHOP in after:
            return np.where(after == -WHITE_BISHOP), np.where(after == WHITE_BISHOP), NO_PROMOTION, NO_REWARD
        if BLACK_KING - WHITE_BISHOP in after:
            return np.where(after == -WHITE_BISHOP), np.where(after == BLACK_KING - WHITE_BISHOP), NO_PROMOTION, WIN_REWARD # white bishop has taken the black king
        else:
            raise Exception("no valid move evaluated for bishop")
    elif -WHITE_KNIGHT in after: # white knight moved
        if -WHITE_KNIGHT in after:
            return np.where(after == -WHITE_KNIGHT), np.where(after == WHITE_KNIGHT), NO_PROMOTION, NO_REWARD
        if BLACK_KING - WHITE_KNIGHT in after:
            return np.where(after == -WHITE_KNIGHT), np.where(after == BLACK_KING - WHITE_KNIGHT), NO_PROMOTION, WIN_REWARD # white knight has taken the black king
        else:
            raise Exception("no valid move evaluated for knight")
    else:
        raise Exception("no valid move evaluated overall")


greedy_policy = { True: 0, False: 0}


def select_action(is_white, is_training, model, episode):
    legal_moves_fen = get_valid_positions(board.fen())
    before = board_to_array(board)
    if not is_white:
        if episode < EPISODES_TRAIN * 0.2:
            result = env.play(board, chess.engine.Limit(time=0.005, depth=1))
        elif episode < EPISODES_TRAIN * 0.4:
            result = env.play(board, chess.engine.Limit(time=0.010, depth=2))
        elif episode < EPISODES_TRAIN * 0.6:
            result = env.play(board, chess.engine.Limit(time=0.020, depth=3))
        elif episode < EPISODES_TRAIN * 0.8:
            result = env.play(board, chess.engine.Limit(time=0.040, depth=4))
        elif episode < EPISODES_TRAIN * 0.9:
            result = env.play(board, chess.engine.Limit(time=0.080, depth=5))
        else:
            result = env.play(board, chess.engine.Limit(time=0.100, depth=6))
        new_board = chess.Board()
        new_board.set_fen(board.fen())
        new_board.push(result.move)
        after = board_to_array(new_board)
        return evaluate_moves(before, after)
    elif random.random() < eps_threshold and is_training:
        greedy_policy[False] += 1
        index = random.randint(0, len(legal_moves_fen)-1)
        after = board_to_array(chess.Board(legal_moves_fen[index]))
        return evaluate_moves(before, after)
    else:
        greedy_policy[True] += 1
        output = q_net.get_q_values(board, model)
        legal_moves_index = np.zeros(BOARD_SIZE)
        move_list = []
        for legal_move in legal_moves_fen:
            after = board_to_array(chess.Board(legal_move))
            move_from, move_to, promotion, reward = evaluate_moves(before, after)
            move_list.append((move_from, move_to, promotion, reward))
            legal_moves_index[move_to[0]] = 1
        output[legal_moves_index != 1] = 0
        output[output == 0] = float('-inf')
        move_to = output.argmax().item()
        for move in move_list:
            if move[1][0].item() == move_to:
                return move[0], move[1], move[2], move[3]
        raise Exception("no valid move found")


promotion_labels = ['','','n','b','r','q']


def get_actions(action_from, action_to, promotion_index):
    action_from = action_from[0].item()
    action_to = action_to[0].item()
    if promotion_index != 0:
        promotion = promotion_labels[promotion_index]
    else:
        promotion = ''
    action_from = letters[action_from % 8] + str(8 - action_from // 8)
    action_to = letters[action_to % 8] + str(8 - action_to // 8)
    return action_from, action_to + promotion


def play_and_print(color, action_from, action_to):
    if SHOW_BOARD:
        print(f"--------------- round: {rounds_training} {color} - san: '{action_from}{action_to}'")
        board_status.cache(board)
    play(action_from, action_to)
    if SHOW_BOARD:
        board_status.print(board)


def play(action_from, action_to):
    move = chess.Move.from_uci(action_from + action_to)
    board.push(move)


def white_is_winner(reason):
    return reason != "Termination.FIVEFOLD_REPETITION" and reason != "Termination.STALEMATE"


def black_is_winner(reason):
    return reason == "Termination.INSUFFICIENT_MATERIAL"


letters = ['a','b','c','d','e','f','g','h']

SHOW_BOARD = False
MAX_ROUNDS = 50
EPISODES_TEST = 1000
EPISODES_TRAIN = 100000

reasons = []

if __name__ == "__main__":

    start_overall = time.time()

    file_name = os.path.join("src", "rl", "chess-statistics-q-net.csv")
    if os.path.isfile(file_name):
        os.remove(file_name)
    with open(file_name, 'a') as f:
        f.write("#of-trainings\twhite-training-won\t#black-training-won\tdraw-training\t#of-test-games\twhite-test-won\tblack-test-won\tdraw-test\tbest-loss\taverage-rounds-played\tduration\n")

    games_training = {"draw": 0, "white": 0, "black": 0}

    print("RL training start ***")

    replay_memory = ReplayMemory(max_length=MAX_MEMORY_SIZE)

    best_loss = float('inf')

    # Keep track of steps since model and target model were updated
    steps_since_model_update = 0

    start_episode = time.time()

    rounds_training_total = 0

    for episode_training in range(1, EPISODES_TRAIN+1):

        board.set_fen("3k4/8/3K4/3P4/8/8/8/8 w - - 0 1")
        rounds_training = 0
        winner_name = "draw"
        global_reward = 0.0

        while not board.is_game_over() and rounds_training < MAX_ROUNDS:
            global_reward -= rounds_training * 0.0001
            action_from_white, action_to_white, promotion, reward_after_white_move = select_action(True, True, q_net.policy_net, episode_training)
            action_from_str, action_to_str = get_actions(action_from_white, action_to_white, promotion)
            state_table_before = board_to_array(board)
            play_and_print("white", action_from_str, action_to_str)
            state_table_after = board_to_array(board)

            if board.is_game_over():
                if reward_after_white_move == NO_REWARD:
                    reward = global_reward
                else:
                    reward = reward_after_white_move
                if SHOW_BOARD:
                    print(f"reward: {reward}, promotion: {promotion}")
                replay_memory.store([state_table_before, action_to_white, reward, state_table_after, True])
                reason = board_status.reason_why_the_game_is_over(board)
                if reason not in reasons:
                    reasons.append(reason)
                if white_is_winner(reason):
                    winner_name = "white"
            else:
                action_from_black, action_to_black, promotion, reward_after_black_move = select_action(False, True, q_net.policy_net, episode_training)
                action_from_str, action_to_str = get_actions(action_from_black, action_to_black, promotion)
                play_and_print("black", action_from_str, action_to_str)
                if board.is_game_over():
                    reason = board_status.reason_why_the_game_is_over(board)
                    if reward_after_black_move == NO_REWARD:
                        reward = global_reward
                    else:
                        reward = reward_after_black_move
                    if SHOW_BOARD:
                        print(f"reward: {reward}")
                    replay_memory.store([state_table_before, action_to_white, reward, state_table_after, True])
                    if reason not in reasons:
                        reasons.append(reason)
                    if black_is_winner(reason):
                        winner_name = "black"
                else:
                    if reward_after_white_move == NO_REWARD and reward_after_black_move == NO_REWARD:
                        reward = global_reward
                    else:
                        reward = reward_after_white_move
                    if SHOW_BOARD:
                        print(f"reward: {reward}, promotion: {promotion}")
                    replay_memory.store([state_table_before, action_to_white, reward, state_table_after, False])

            rounds_training += 1
            steps_since_model_update += 1

        rounds_training_total += rounds_training

        if steps_since_model_update >= 10:
            loss = q_net.optimize(replay_memory)
            steps_since_model_update = 0
            if loss is not None and loss < best_loss:
                best_loss = loss
                # Soft update of the target network's weights
                target_net_state_dict = q_net.target_net.state_dict()
                policy_net_state_dict = q_net.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
                q_net.target_net.load_state_dict(target_net_state_dict)

        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * rounds_training_total / EPS_DECAY)
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
                    action_from_white, action_to_white, promotion, reward_after_white_move = select_action(True, False, model, EPISODES_TRAIN)
                    action_from_str, action_to_str = get_actions(action_from_white, action_to_white, promotion)
                    play(action_from_str, action_to_str)

                    if board.is_game_over():
                        reason = board_status.reason_why_the_game_is_over(board)
                        if reason not in reasons:
                            reasons.append(reason)
                        if white_is_winner(reason):
                            winner_name = "white"
                    else:
                        action_from_white, action_to_white, promotion, reward_after_white_move = select_action(False,False, model, EPISODES_TRAIN)
                        action_from_str, action_to_str = get_actions(action_from_white, action_to_white, promotion)
                        play(action_from_str, action_to_str)
                        if board.is_game_over():
                            reason = board_status.reason_why_the_game_is_over(board)
                            if reason not in reasons:
                                reasons.append(reason)
                            if black_is_winner(reason):
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
    print(f"exploration:  {greedy_policy[False]}")
    print(f"exploitation: {greedy_policy[True]}")

    print("-----------------------------")
    print("RL training end ***")

    env.close()
    end = time.time()
    with open(file_name, 'a') as f:
        f.write(f"duration: {(end - start_overall):0.1f}s\n")
    print(f"duration: {(end - start_overall):0.1f}s")
