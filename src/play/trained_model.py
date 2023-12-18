import os

import torch

from src.lib.utilities import get_device
from src.model.ChessmaitCnn2NonHot import ChessmaitCnn2NonHot
from src.model.ChessmaitCnn4Bitboard import ChessmaitCnn4Bitboard
from src.model.ChessmaitMlp1 import ChessmaitMlp1
from src.model.ChessmaitMlp4 import ChessmaitMlp4
from src.model.ChessmaitMlp5 import ChessmaitMlp5

PATH_TO_MODEL = os.path.join("models")


class TrainedModel:
    def __init__(self, file_name, model_to_use, fen_to_tensor, _device, use_normalization=False, max_evaluation=0,
                 min_evaluation=0):
        self.file_name = file_name
        self.fen_to_tensor = fen_to_tensor
        self.min_evaluation = min_evaluation
        self.max_evaluation = max_evaluation
        self.use_normalization = use_normalization
        if model_to_use == 'ChessmaitMlp5':
            self.model = ChessmaitMlp5()
        elif model_to_use == 'ChessmaitMlp4':
            self.model = ChessmaitMlp4()
        elif model_to_use == 'ChessmaitCnn2NonHot':
            self.model = ChessmaitCnn2NonHot()
        elif model_to_use == 'ChessmaitCnn4Bitboard':
            self.model = ChessmaitCnn4Bitboard()
        elif model_to_use == 'ChessmaitMlp1':
            self.model = ChessmaitMlp1()
        else:
            raise Exception('Unknown model')
        self.model.load_state_dict(torch.load(os.path.join(PATH_TO_MODEL, self.file_name), map_location=_device))
        self.model.eval()

    def de_normalize(self, normalized_evaluation):
        if self.use_normalization:
            return normalized_evaluation * (self.max_evaluation - self.min_evaluation) + self.min_evaluation
        return normalized_evaluation

    def get_name(self):
        return self.file_name[:-4]


device = get_device()
model_wild_snow_28 = TrainedModel("wild-snow-28.pth", 'ChessmaitMlp1', 'fen_to_tensor_one_board', device,
                                  use_normalization=True, max_evaluation=20000, min_evaluation=-20000)
model_fluent_mountain_47 = TrainedModel("fluent-mountain-47.pth", 'ChessmaitMlp5', 'fen_to_tensor_one_board', device,
                                        use_normalization=True, max_evaluation=14105, min_evaluation=-14905)
model_divine_leaf_29 = TrainedModel("divine-leaf-29.pth", 'ChessmaitMlp4', 'fen_to_tensor_one_board', device,
                                    use_normalization=True, max_evaluation=12352, min_evaluation=-12349)
model_honest_dragon_72 = TrainedModel("honest-dragon-72.pth", 'ChessmaitCnn2NonHot', 'fen_to_cnn_tensor_non_hot_enc',
                                      device, use_normalization=True, max_evaluation=15265, min_evaluation=-15265)
model_upbeat_cloud_79 = TrainedModel("upbeat-cloud-79.pth", 'ChessmaitMlp5', 'fen_to_tensor_one_board', device)

# models for contest
model_smart_valley_6 = TrainedModel("smart-valley-6.pth", 'ChessmaitMlp1', 'fen_to_tensor_one_board', device,
                                    use_normalization=True, max_evaluation=20000, min_evaluation=-20000)
model_firm_star_24 = TrainedModel("firm-star-24.pth", 'ChessmaitMlp1', 'fen_to_tensor_one_board', device,
                                  use_normalization=True, max_evaluation=12352, min_evaluation=-12349)
model_lemon_plasma_103 = TrainedModel("lemon-plasma-103.pth", 'ChessmaitMlp5', 'fen_to_tensor_one_board', device)
model_stellar_sound_105 = TrainedModel("stellar-sound-105.pth", 'ChessmaitMlp5', 'fen_to_tensor_one_board', device)
model_effortless_vortex_142 = TrainedModel("effortless-vortex-142.pth", 'ChessmaitCnn4Bitboard', 'fen_to_bitboard',
                                           device)
model_graceful_glitter_166 = TrainedModel("graceful-glitter-166.pth", 'ChessmaitMlp5', 'fen_to_tensor_one_board',
                                          device)
model_apricot_armadillo_167 = TrainedModel("apricot-armadillo-167.pth", 'ChessmaitMlp5', 'fen_to_tensor_one_board',
                                           device)
model_fresh_blaze_174 = TrainedModel("fresh-blaze-174.pth", 'ChessmaitCnn4Bitboard', 'fen_to_bitboard', device)
