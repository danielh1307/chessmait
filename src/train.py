import argparse
import os
from typing import Union

import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from loss.CustomWeightedMSELoss import CustomWeightedMSELoss
from src.PositionToEvaluationDataset import PositionToEvaluationDataset
from src.PositionToEvaluationDatasetClassification import PositionToEvaluationDatasetClassification
from src.lib.utilities import get_device
from src.model.ChessmaitCnn4Bitboard import ChessmaitCnn4Bitboard


############################################################
# This is the central Python script to perform the training
############################################################

def get_training_configuration() -> argparse.Namespace:
    """
    Sets configuration values for this training.

    Returns
    -------
    argparse.Namespace
        the namespace with configuration values for this training.

    """
    _config = argparse.Namespace()
    _config.train_percentage = 0.85  # percentage of data which is used for training
    _config.learning_rate = 0.0001

    # for Adam optimizer
    _config.betas = (0.90, 0.99)  # needed for Adam optimizer
    _config.eps = 1e-8  # needed for Adam optimizer

    # for SGD optimizer
    # _config.momentum = 0.7
    # _config.weight_decay = 1e-8

    _config.num_workers = 20
    _config.epochs = 50
    _config.batch_size = 64
    _config.fen_to_tensor_method = FEN_TO_TENSOR_METHOD

    return _config


#################################################################################################
# Make sure to set those values correctly before starting your training
#################################################################################################
# Datafiles to load
PATH_TO_DATAFILE = os.path.join("data", "preprocessed")
PATH_TO_PICKLEFILE = os.path.join("data", "angelo")
# set either PICKLE_FILES or DATA_FILES
# PICKLE_FILES: contains already the correct tensors
# DATA_FILES: contains FEN position, tensors have to be created
PICKLE_FILES = ["01_with_mate_bitboard.pkl", "02_with_mate_bitboard.pkl", "03_with_mate_bitboard.pkl",
                "04_with_mate_bitboard.pkl", "06_with_mate_bitboard.pkl",
                "07_with_mate_bitboard.pkl", "08_with_mate_bitboard.pkl", "09_with_mate_bitboard.pkl",
                "10_with_mate_bitboard.pkl"]
DATA_FILES = []

WANDB_REPORTING = True
REGRESSION_TRAINING = True
FEN_TO_TENSOR_METHOD = "fen_to_bitboard"  # just for documentation

# Model, loss function, optimizer
config = get_training_configuration()
model = ChessmaitCnn4Bitboard()
loss_function = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.betas, eps=config.eps)


#################################################################################################


def get_dataloaders(_config: argparse.Namespace) -> (DataLoader, DataLoader, DataLoader):
    """

    Parameters
    ----------
    _config : argparse.Namespace
        configuration values for this training

    Returns
    -------
    tuple
        Dataloader for training, validation and testing

    """
    print("Prepare dataloaders ...")
    csv_files = []
    pickle_files = []
    for data_file in DATA_FILES:
        csv_files.append(os.path.join(PATH_TO_DATAFILE, data_file))

    for data_file in PICKLE_FILES:
        pickle_files.append(os.path.join(PATH_TO_PICKLEFILE, data_file))

    if REGRESSION_TRAINING:
        dataset = PositionToEvaluationDataset(csv_files, pickle_files)
        _config.min_evaluation, _config.max_evaluation = dataset.get_min_max_score()
        print(f"Min score is {_config.min_evaluation} and max score is {_config.max_evaluation}")
    else:
        dataset = PositionToEvaluationDatasetClassification(csv_files, pickle_files)

    torch.manual_seed(42)

    train_size = int(_config.train_percentage * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    _config.number_of_evaluations_for_training = train_size
    print(f"Number of training games is {_config.number_of_evaluations_for_training}")

    # Create DataLoaders
    _train_loader = DataLoader(train_dataset, batch_size=_config.batch_size, shuffle=True,
                               num_workers=_config.num_workers)
    _val_loader = DataLoader(val_dataset, batch_size=_config.batch_size, shuffle=False, num_workers=_config.num_workers)

    print("Prepare dataloaders finished ...")
    return _train_loader, _val_loader


def train_model(_config: argparse.Namespace,
                _model: nn.Module,
                _optimizer: Union[torch.optim.Adam, torch.optim.SGD],
                _scheduler,
                _loss_function: Union[nn.CrossEntropyLoss, nn.MSELoss, CustomWeightedMSELoss],
                _train_loader: DataLoader,
                _val_loader: DataLoader,
                _device: str,
                _model_name: str):
    print("Starting training ...")
    _model.to(_device)

    if WANDB_REPORTING:
        wandb.watch(_model)

    best_val_loss = float('inf')

    for epoch in range(1, _config.epochs + 1):
        # Training
        _model.train()
        train_loss = 0.0

        batch_number = 0
        for position, evaluation in _train_loader:
            if batch_number % 1000 == 0:
                print(f"batch {batch_number} from {len(_train_loader)} ...")
            batch_number += 1
            _optimizer.zero_grad()
            predicted_evaluation = _model(position.to(_device))

            if REGRESSION_TRAINING:
                evaluation = evaluation.unsqueeze(1)  # Reshapes to match the predicted_evaluation
            loss = _loss_function(predicted_evaluation, evaluation.to(_device))
            loss.backward()
            _optimizer.step()
            train_loss += loss.item() * position.size(0)

        # Validation
        _model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for position, evaluation in _val_loader:
                predicted_evaluation = _model(position.to(_device))

                if REGRESSION_TRAINING:
                    evaluation = evaluation.unsqueeze(1)  # Reshapes to match the predicted_evaluation
                loss = _loss_function(predicted_evaluation, evaluation.to(_device))
                val_loss += loss.item() * position.size(0)

        # calculate average losses
        train_loss = train_loss / len(_train_loader.dataset)
        val_loss = val_loss / len(_val_loader.dataset)

        epoch_result = {"epoch": epoch,
                        "training loss": train_loss,
                        "validation loss": val_loss
                        }
        print(epoch_result)
        if WANDB_REPORTING:
            wandb.log(epoch_result)

        # not sure if this really has any benefit - for the moment, we comment it out
        # step the scheduler - adjust the learning rate if validation loss stops decreasing
        # _scheduler.step(val_loss)

        # Save model if validation loss has decreased
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), _model_name)
            best_val_loss = val_loss

    print("Training finished ...")


if __name__ == "__main__":
    print("Starting training process ...")
    print(f"WANDB_REPORTING for this training is set to {WANDB_REPORTING} ...")
    if REGRESSION_TRAINING:
        print("Training on a regression problem ...")
    else:
        print("Training on a classification problem ...")

    print("Training on files " + str(DATA_FILES))

    device = get_device()

    train_loader, val_loader = get_dataloaders(config)

    model_name = "best_model.pth"
    if WANDB_REPORTING:
        config.model = type(model).__name__
        config.loss_function = type(loss_function).__name__
        wand_return = wandb.init(project="chessmait", config=vars(config))
        model_name = wand_return.name + ".pth"

    # adding a scheduler to reduce the learning_rate as soon as the validation loss stops decreasing,
    # this is to try to prevent overfitting of the model
    scheduler = ReduceLROnPlateau(optimizer, 'min')  # 'min' means reducing the LR when the metric stops decreasing

    train_model(config, model, optimizer, scheduler, loss_function, train_loader, val_loader, device, model_name)
    wandb.finish()
