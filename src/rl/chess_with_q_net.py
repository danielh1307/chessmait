# inspired by https://nestedsoftware.com/2019/12/27/tic-tac-toe-with-a-neural-network-1fjn.206436.html
# and https://medium.com/towards-data-science/hands-on-deep-q-learning-9073040ce841

import copy
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import chess.engine
from collections import deque, namedtuple
from src.lib.position_validator import get_valid_positions
from src.board_status import BoardStatus

PATH_TO_CHESS_ENGINE = os.path.join("stockfish", "stockfish-windows-x86-64-avx2.exe")

EPS_DECAY = 0.95 # eps gets multiplied by this number each epoch...
MIN_EPS = 0.1 # ...until this minimum eps is reached
GAMMA = 0.95 # discount
MAX_MEMORY_SIZE = 10000 # size of the replay memory
BATCH_SIZE = 32 # batch size of the neural network training

POSSIBLE_INDEXES = np.arange(9)
BOARD_SIZE = 64
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
        self.policy_net = self.policy_net.train()
        self.target_net = QNet().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net = self.target_net.eval()

        self.optimizer = optim.SGD(self.policy_net.parameters(), lr=1e-4)
        self.loss_function = nn.MSELoss()

    def optimize(self, replay_memory, batch_size):
        if len(replay_memory) < BATCH_SIZE:
            return

        mini_batch = replay_memory.structured_sample(batch_size) # get samples from the replay memory

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
                max_future_q = reward + GAMMA * torch.max(target_qs[index])
            else:
                # If terminal, only include the immediate reward
                max_future_q = reward

            # The Qs for this sample of the mini batch
            updated_qs_sample = initial_qs[index]
            # Update the value for the taken action
            action_to = action[1][0].item()
            updated_qs_sample[action_to] = max_future_q

            # Keep track of the observation and updated Q value
            states[index] = observation
            updated_qs[index] = updated_qs_sample

        predicted_qs = self.policy_net(torch.tensor(states, device=device, dtype=torch.double))
        loss = self.loss_function(predicted_qs, updated_qs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
#env = chess.engine.SimpleEngine.popen_uci(PATH_TO_CHESS_ENGINE)
#env.configure({"UCI_Elo": 1320})
board = chess.Board()
board_status = BoardStatus()
eps = 0.3 # exploration rate, probability of choosing random action


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


def evaluate_moves(before, after):
    before[before == -6] = BLACK_KING
    after[after == -6] = BLACK_KING
    after -= before
    if -BLACK_KING in after: # black king moved
        if BLACK_KING in after:
            return np.where(after == -BLACK_KING), np.where(after == BLACK_KING), 0
        if BLACK_KING - WHITE_PAWN in after:
            return np.where(after == -BLACK_KING), np.where(after == BLACK_KING - WHITE_PAWN), 0 # black king has taken the white pawn
        if BLACK_KING - WHITE_KING in after:
            return np.where(after == -BLACK_KING), np.where(after == BLACK_KING - WHITE_KING), 0 # black king has taken the white king
        if BLACK_KING - WHITE_QUEEN in after:
            return np.where(after == -BLACK_KING), np.where(after == BLACK_KING - WHITE_QUEEN), 0 # black king has taken the white queen
        if BLACK_KING - WHITE_ROOK in after:
            return np.where(after == -BLACK_KING), np.where(after == BLACK_KING - WHITE_ROOK), 0 # black king has taken the white rook
        if BLACK_KING - WHITE_BISHOP in after:
            return np.where(after == -BLACK_KING), np.where(after == BLACK_KING - WHITE_BISHOP), 0 # black king has taken the white bishop
        if BLACK_KING - WHITE_KNIGHT in after:
            return np.where(after == -BLACK_KING), np.where(after == BLACK_KING - WHITE_KNIGHT), 0 # black king has taken the white bishop
        else:
            raise Exception("no valid move evaluated for black king")
    elif -WHITE_KING in after:  # white king moved
        if WHITE_KING in after:
            return np.where(after == -WHITE_KING), np.where(after == WHITE_KING), 0
        if BLACK_KING - WHITE_KING in after:
            return np.where(after == -WHITE_KING), np.where(after == BLACK_KING - WHITE_KING), 0 # white king has taken the black king
        else:
            raise Exception("no valid move evaluated for white king")
    elif -WHITE_PAWN in after: # white pawn moved
        if WHITE_PAWN in after:
            return np.where(after == -WHITE_PAWN), np.where(after == WHITE_PAWN), 0
        elif WHITE_KNIGHT in after:
            return np.where(after == -WHITE_PAWN), np.where(after == WHITE_KNIGHT), WHITE_KNIGHT # convert pawn to knight
        elif WHITE_BISHOP in after:
            return np.where(after == -WHITE_PAWN), np.where(after == WHITE_BISHOP), WHITE_BISHOP # convert pawn to bishop
        elif WHITE_ROOK in after:
            return np.where(after == -WHITE_PAWN), np.where(after == WHITE_ROOK), WHITE_ROOK # convert pawn to rook
        elif WHITE_QUEEN in after:
            return np.where(after == -WHITE_PAWN), np.where(after == WHITE_QUEEN), WHITE_QUEEN # convert pawn to queen
        elif BLACK_KING - WHITE_PAWN in after:
            return np.where(after == -WHITE_PAWN), np.where(after == BLACK_KING - WHITE_PAWN), 5 # white pawn has taken the black king
        else:
            raise Exception("no valid move evaluated for pawn")
    elif -WHITE_QUEEN in after: # white queen moved
        if -WHITE_QUEEN in after:
            return np.where(after == -WHITE_QUEEN), np.where(after == WHITE_QUEEN), 0
        if BLACK_KING - WHITE_QUEEN in after:
            return np.where(after == -WHITE_QUEEN), np.where(after == BLACK_KING - WHITE_QUEEN), 0
        else:
            raise Exception("no valid move evaluated for queen")
    elif -WHITE_ROOK in after: # white rook moved
        if -WHITE_ROOK in after:
            return np.where(after == -WHITE_ROOK), np.where(after == WHITE_ROOK), 0
        if BLACK_KING - WHITE_ROOK in after:
            return np.where(after == -WHITE_ROOK), np.where(after == BLACK_KING - WHITE_ROOK), 0
        else:
            raise Exception("no valid move evaluated for rook")
    elif -WHITE_BISHOP in after: # white bishop moved
        if -WHITE_BISHOP in after:
            return np.where(after == -WHITE_BISHOP), np.where(after == WHITE_BISHOP), 0
        if BLACK_KING - WHITE_BISHOP in after:
            return np.where(after == -WHITE_BISHOP), np.where(after == BLACK_KING - WHITE_BISHOP), 0
        else:
            raise Exception("no valid move evaluated for bishop")
    elif -WHITE_KNIGHT in after: # white knight moved
        if -WHITE_KNIGHT in after:
            return np.where(after == -WHITE_KNIGHT), np.where(after == WHITE_KNIGHT), 0
        if BLACK_KING - WHITE_KNIGHT in after:
            return np.where(after == -WHITE_KNIGHT), np.where(after == BLACK_KING - WHITE_KNIGHT), 0
        else:
            raise Exception("no valid move evaluated for knight")
    else:
        raise Exception("no valid move evaluated overall")


def select_action(is_white, is_training, model):
    legal_moves_fen = get_valid_positions(board.fen())
    before = board_to_array(board)
    if not is_white or random.random() < eps and is_training:
        index = random.randint(0, len(legal_moves_fen)-1)
        after = board_to_array(chess.Board(legal_moves_fen[index]))
        return evaluate_moves(before, after)
    else:
        output = q_net.get_q_values(board, q_net.policy_net)
        legal_moves_index = np.zeros(BOARD_SIZE)
        move_list = []
        for legal_move in legal_moves_fen:
            after = board_to_array(chess.Board(legal_move))
            move_from, move_to, conversion = evaluate_moves(before, after)
            move_list.append((move_from, move_to, conversion))
            legal_moves_index[move_to[0]] = 1
        output[legal_moves_index != 1] = 0
        move_to = output.argmax().item()
        for move in move_list:
            if move[1][0].item() == move_to:
                return move[0], move[1], move[2]
        raise Exception("no valid move found")


promotion_labels = ['','','n','b','r','q']


def get_actions(action):
    a_from = action[0][0].item()
    a_to = action[1][0].item()
    promotion_index = action[2]
    if promotion_index != 0:
        promotion = promotion_labels[promotion_index]
    else:
        promotion = ''
    action_from = letters[a_from % 8] + str(8 - a_from // 8)
    action_to = letters[a_to % 8] + str(8 - a_to // 8)
    return action_from, action_to + promotion


def play_and_print(color, action_from, action_to):
    print(f"--------------- round: {rounds} {color} - san: '{action_from}{action_to}'")
    board_status.cache(board)
    play(action_from, action_to)
    board_status.print(board)


def play(action_from, action_to):
    move = chess.Move.from_uci(action_from + action_to)
    board.push(move)


letters = ['a','b','c','d','e','f','g','h']


if __name__ == "__main__":

    start = time.time()

    file_name = "chess-statistics-q-net.csv"
    if os.path.isfile(file_name):
        os.remove(file_name)
    with open(file_name, 'a') as f:
        f.write("#of-trainings\twhite-training-won\t#black-training-won\tdraw-training\t#of-test-games\twhite-test-won\tblack-test-won\tdraw-test\tbest-loss\n")

    games_training = {"draw": 0, "white": 0, "black": 0}

    print("RL training start ***")

    replay_memory = ReplayMemory(max_length=MAX_MEMORY_SIZE)

    best_loss = float('inf')

    # Keep track of steps since model and target model were updated
    steps_since_model_update = 0

    for episode_training in range(101):

        board.set_fen("3k4/8/3K4/3P4/8/8/8/8 w - - 0 1")
        rounds = 0
        winner_name = "draw"
        reward = 0.0

        while not board.is_game_over() and rounds < 50:
            reward -= rounds * 0.001
            action = select_action(True, True, q_net.policy_net)
            action_from, action_to = get_actions(action)
            state_table_before = board_to_array(board)
            play_and_print("white", action_from, action_to)

            if board.is_game_over():
                reward = 1.0 if board.is_checkmate() or action[2] > 0 else reward
                print(f"reward: {reward}, promotion: {action[2]}")
                state_table_after = board_to_array(board)
                replay_memory.store([state_table_before, action, reward, state_table_after, True])
                winner_name = "white"
            else:
                reward = 1.0 if action[2] > 0 else reward
                action = select_action(False, True, q_net.policy_net)
                action_from, action_to = get_actions(action)
                play_and_print("black", action_from, action_to)
                state_table_after = board_to_array(board)
                if board.is_game_over():
                    reward = -1.0 if board.is_checkmate() else reward
                    print(f"reward: {reward}, promotion: {action[2]}")
                    replay_memory.store([state_table_before, action, reward, state_table_after, True])
                    winner_name = "black"
                else:
                    print(f"reward: {reward}, promotion: {action[2]}")
                    replay_memory.store([state_table_before, action, reward, state_table_after, False])

            rounds += 1
            steps_since_model_update += 1

        if steps_since_model_update >= 10:
            loss = q_net.optimize(replay_memory, BATCH_SIZE)
            steps_since_model_update = 0
            if loss is not None and loss < best_loss:
                best_loss = loss
                q_net.target_net.load_state_dict(q_net.policy_net.state_dict())

        eps = max(MIN_EPS, eps * EPS_DECAY)
        games_training[winner_name] += 1

        if episode_training % 10 == 0 and episode_training != 0:
            print(f"episode: {episode_training} - best-loss: {best_loss}")

        if episode_training % 100 == 0 and episode_training != 0:

            print("-----------------------------")
            print(f"Tests episode: {episode_training}")

            games_test = {"draw": 0, "white": 0, "black": 0}

            model = copy.deepcopy(q_net.target_net)
            model.eval()

            for episode_test in range(1000):

                board.set_fen("3k4/8/3K4/3P4/8/8/8/8 w - - 0 1")
                rounds_test = 0
                winner_name = "draw"

                while not board.is_game_over():
                    action = select_action(True, False, model)
                    action_from, action_to = get_actions(action)
                    play(action_from, action_to)

                    if board.is_game_over():
                        winner_name = "white"
                    else:
                        action = select_action(False, False, model)
                        action_from, action_to = get_actions(action)
                        play(action_from, action_to)
                        if board.is_game_over():
                            winner_name = "black"

                    rounds_test += 1

                games_test[winner_name] += 1

            print("Player     Training     Test")
            print(f"draw:      {games_training['draw']:8}\t{games_test['draw']:8}")
            print(f"white:     {games_training['white']:8}\t{games_test['white']:8}")
            print(f"black:     {games_training['black']:8}\t{games_test['black']:8}")

            with open(file_name, 'a') as f:
                f.write(f"{episode_training}\t{games_training['white']}\t{games_training['black']}\t{games_training['draw']}\t1000\t{games_test['white']}\t{games_test['black']}\t{games_test['draw']}\t{best_loss}\n")

    torch.save(q_net.target_net.state_dict(), "chess.pth")

    print("-----------------------------")
    print("RL training end ***")

    #env.close()
    end = time.time()
    with open(file_name, 'a') as f:
        f.write(f"duration: {(end - start):0.1f}s\n")
    print(f"duration: {(end - start):0.1f}s")
