import torch
import os
from src.model.ChessmaitMlp1 import ChessmaitMlp1
from src.model.ChessmaitMlp4 import ChessmaitMlp4
from src.model.ChessmaitMlp5 import ChessmaitMlp5
from src.model.ChessmaitCnn2NonHot import ChessmaitCnn2NonHot
from src.train import get_device

PATH_TO_MODEL = os.path.join("models")


class TrainedModel:
    def __init__(self, file_name, max_evaluation, min_evaluation, version, fen_to_tensor, use_normalization=True):
        self.file_name = file_name
        self.fen_to_tensor = fen_to_tensor
        self.min_evaluation = min_evaluation
        self.max_evaluation = max_evaluation
        self.use_normalization = use_normalization
        if version == 2:
            self.model = ChessmaitMlp5()
        elif version == 3:
            self.model = ChessmaitMlp4()
        elif version == 4:
            self.model = ChessmaitCnn2NonHot()
        else:
            self.model = ChessmaitMlp1()
        device = get_device(False)
        self.model.load_state_dict(torch.load(os.path.join(PATH_TO_MODEL, self.file_name), map_location=device))
        self.model.eval()

    def de_normalize(self, normalized_evaluation):
        # normalized_evaluation = (evaluation - MIN_EVALUATION) / (MAX_EVALUATION - MIN_EVALUATION)
        if self.use_normalization:
            return normalized_evaluation * (self.max_evaluation - self.min_evaluation) + self.min_evaluation
        return normalized_evaluation


model_firm_star_24 = TrainedModel("firm-star-24.pth", 12352, -12349, 1, 0)
model_smart_valley_6 = TrainedModel("smart-valley-6.pth", 20000, -20000, 1, 0)
model_wild_snow_28 = TrainedModel("wild-snow-28.pth", 20000, -20000, 1, 0)
model_fluent_mountain_47 = TrainedModel("fluent-mountain-47.pth", 14105, -14905, 2, 0)
model_divine_leaf_29 = TrainedModel("divine-leaf-29.pth", 12352, -12349, 3, 0)
model_honest_dragon_72 = TrainedModel("honest-dragon-72.pth", 15265, -15265, 4, 1)
model_upbeat_cloud_79 = TrainedModel("upbeat-cloud-79.pth", 1000, -1000, 2, 0, use_normalization=False)

