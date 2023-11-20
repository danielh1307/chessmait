import torch
import os
from src.model.ChessmaitMlp1 import ChessmaitMlp1

PATH_TO_MODEL = os.path.join("models")


class TrainedModel:
    def __init__(self, file_name, max_evaluation, min_evaluation):
        self.file_name = file_name
        self.min_evaluation = min_evaluation
        self.max_evaluation = max_evaluation
        self.model = ChessmaitMlp1()
        self.model.load_state_dict(torch.load(os.path.join(PATH_TO_MODEL, self.file_name)))
        self.model.eval()

    def de_normalize(self, normalized_evaluation):
        # normalized_evaluation = (evaluation - MIN_EVALUATION) / (MAX_EVALUATION - MIN_EVALUATION)
        return normalized_evaluation * (self.max_evaluation - self.min_evaluation) + self.min_evaluation


model_firm_star_24 = TrainedModel("firm-star-24.pth", 12352, -12349)
model_smart_valley_6 = TrainedModel("smart-valley-6.pth", 12352, -12349)
model_wild_snow_28 = TrainedModel("wild-snow-28.pth", 20000, -20000)

