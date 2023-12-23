import os
import numpy as np
import torch
import torch.nn as nn
import random
import chess.engine
from collections import deque
from src.lib.utilities import get_valid_positions
from src.play.board_status import BoardStatus

PATH_TO_CHESS_ENGINE = os.path.join("stockfish", "stockfish-windows-x86-64-avx2.exe")

BATCH_SIZE = 32 # the number of transitions sampled from the replay buffer
GAMMA = 0.99 # the discount factor as mentioned in the previous section
EPS_START = 0.9 # the starting value of epsilon
EPS_END = 0.2 # the final value of epsilon
EPS_DECAY = 1000 # controls the rate of exponential decay of epsilon, higher means a slower decay
LR = 0.0001 # the learning rate of the optimizer
MAX_MEMORY_SIZE = 10000 # size of the replay memory

POSSIBLE_INDEXES = np.arange(9)
BOARD_SIZE = 64
OUT_FEATURES = 4096
SHOW_BOARD = False

promotion_labels = ['','','n','b','r','q']
letters = ['a','b','c','d','e','f','g','h']

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
HALF_REWARD = 0.5

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
board_status = BoardStatus()

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(BOARD_SIZE, OUT_FEATURES),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(OUT_FEATURES, OUT_FEATURES),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(OUT_FEATURES, OUT_FEATURES),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(OUT_FEATURES, BOARD_SIZE),
        )
        self.double()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output_layer(x)

def board_to_tensor_with_state(this_state):
    return torch.tensor(this_state, device=device, dtype=float).flatten()


def board_to_tensor_with_board(this_board):
    int_board = board_status.convert_to_int(this_board)
    return torch.tensor(np.array(int_board), device=device, dtype=float).flatten()


def board_to_array(this_board):
    return np.array(board_status.convert_to_int(this_board)).flatten()

def get_q_values(board, model):
    inputs = board_to_tensor_with_board(board)
    outputs = model(inputs)
    return outputs


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
            return np.where(after == -WHITE_PAWN), np.where(after == WHITE_KNIGHT), WHITE_KNIGHT, HALF_REWARD # convert pawn to knight
        elif WHITE_BISHOP in after:
            return np.where(after == -WHITE_PAWN), np.where(after == WHITE_BISHOP), WHITE_BISHOP, HALF_REWARD # convert pawn to bishop
        elif WHITE_ROOK in after:
            return np.where(after == -WHITE_PAWN), np.where(after == WHITE_ROOK), WHITE_ROOK, HALF_REWARD # convert pawn to rook
        elif WHITE_QUEEN in after:
            return np.where(after == -WHITE_PAWN), np.where(after == WHITE_QUEEN), WHITE_QUEEN, HALF_REWARD # convert pawn to queen
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


def select_action(env, board, eps_threshold, is_white, model, is_training):
    legal_moves_fen = get_valid_positions(board.fen())
    before = board_to_array(board)
    if is_training and random.random() < eps_threshold:
        index = random.randint(0, len(legal_moves_fen) - 1)
        after = board_to_array(chess.Board(legal_moves_fen[index]))
        return evaluate_moves(before, after)
    elif not is_white:
        depth = random.randint(1,5)
        result = env.play(board, chess.engine.Limit(time=depth / 100, depth=depth))
        new_board = chess.Board()
        new_board.set_fen(board.fen())
        new_board.push(result.move)
        after = board_to_array(new_board)
        return evaluate_moves(before, after)
    else:
        output = get_q_values(board, model)
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
        # take the second-highest value to add a bit of randomness, otherwise the same moves will always be played
        if random.randint(0, 1) == 0:
            output[move_to] = float('-inf')
            move_to = output.argmax().item()

        for move in move_list:
            if move[1][0].item() == move_to:
                return move[0], move[1], move[2], move[3]
        raise Exception("no valid move found")


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


def play_and_print(board, rounds, color, action_from, action_to):
    if SHOW_BOARD:
        print(f"--------------- round: {rounds} {color} - san: '{action_from}{action_to}'")
        board_status.cache(board)
    play(board, action_from, action_to)
    if SHOW_BOARD:
        board_status.print(board)


def play(board, action_from, action_to):
    move = chess.Move.from_uci(action_from + action_to)
    board.push(move)


def white_is_winner(reason):
    return reason != "Termination.FIVEFOLD_REPETITION" \
        and reason != "Termination.STALEMATE" \
        and reason != "Termination.INSUFFICIENT_MATERIAL"


def black_is_winner(reason):
    return reason == "Termination.INSUFFICIENT_MATERIAL"
