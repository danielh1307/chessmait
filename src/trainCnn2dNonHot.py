import argparse
import os
import fnmatch
from typing import Union

import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from src.PositionToEvaluationDatasetConv2dNonHot import PositionToEvaluationDatasetConv2dNonHot
from src.model.ChessmaitCnn2NonHot import ChessmaitCnn2NonHot

############################################################
# This is the central Python script to perform the training
############################################################

PATH_TO_DATAFILE = os.path.join("data", "preprocessed")
PATH_TO_PICKLEFILE = os.path.join("data", "pickle")

############################################################################
# Make sure these parameters are correctly set before you start the training
############################################################################
DATA_FILES = []
PICKLE_FILES = []
matching_files = [file for file in os.listdir(PATH_TO_PICKLEFILE) if
                  fnmatch.fnmatch(file, "*.pkl")]
file_names = [os.path.basename(file) for file in matching_files]
for file_name in file_names:
    PICKLE_FILES.append(file_name)

WANDB_REPORTING = True
REGRESSION_TRAINING = True
FEN_TO_TENSOR_METHOD = "fen_to_cnn_tensor_non_hot_enc"

model = ChessmaitCnn2NonHot()



def get_device():
    """
    Checks which device is most appropriate to perform the training.
    If cuda is available, cuda is returned, otherwise mps or cpu.

    Returns
    -------
    str
        the device which is used to perform the training.

    """
    _device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"For this training, we are going to use {_device} device ...")
    return _device


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
    _config.learning_rate = 0.001 #0.001
    _config.betas = (0.90, 0.99)  # needed for Adam optimizer
    _config.eps = 1e-8  # needed for Adam optimizer
    _config.epochs = 15
    _config.batch_size = 256
    _config.fen_to_tensor_method = FEN_TO_TENSOR_METHOD

    return _config


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

    dataset = PositionToEvaluationDatasetConv2dNonHot(csv_files, pickle_files)
    _config.min_evaluation, _config.max_evaluation = dataset.get_min_max_score()
    print(f"Min score is {_config.min_evaluation} and max score is {_config.max_evaluation}")

    torch.manual_seed(42)

    train_size = int(_config.train_percentage * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    _config.number_of_evaluations_for_training = train_size
    print(f"Number of training games is {_config.number_of_evaluations_for_training}")

    # Create DataLoaders
    _train_loader = DataLoader(train_dataset, batch_size=_config.batch_size, shuffle=True)
    _val_loader = DataLoader(val_dataset, batch_size=_config.batch_size, shuffle=False)

    print("Prepare dataloaders finished ...")
    return _train_loader, _val_loader


def train_model(_config: argparse.Namespace,
                _model: nn.Module,
                _optimizer: torch.optim.Adam,
                _scheduler,
                _loss_function: Union[nn.CrossEntropyLoss, nn.MSELoss, nn.HuberLoss],
                _train_loader: DataLoader,
                _val_loader: DataLoader,
                _device: str):
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
                print(f"batch {batch_number} from {len(train_loader)} ...")
            batch_number += 1
            _optimizer.zero_grad()
            predicted_evaluation = _model(position.to(_device))

            if REGRESSION_TRAINING:
                evaluation = evaluation.unsqueeze(1)  # Reshapes [64] to [64, 1] to match the predicted_evaluation
            loss = _loss_function(predicted_evaluation, evaluation.to(_device))
            loss.backward()
            _optimizer.step()
            train_loss += loss.item() * position.size(0)

        # Validation
        _model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for position,evaluation in _val_loader:
                predicted_evaluation = _model(position.to(_device))

                if REGRESSION_TRAINING:
                    evaluation = evaluation.unsqueeze(1)  # Reshapes to match the predicted_evaluation
                loss = _loss_function(predicted_evaluation,evaluation.to(_device))
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
            torch.save(model.state_dict(), "best_model.pth")
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
    config = get_training_configuration()
    train_loader, val_loader = get_dataloaders(config)

    if REGRESSION_TRAINING:
        loss_function = nn.HuberLoss()
    else:
        loss_function = nn.CrossEntropyLoss()

    if WANDB_REPORTING:
        config.model = type(model).__name__
        config.loss_function = type(loss_function).__name__
        wandb.init(project="chessmait", config=vars(config))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.betas, eps=config.eps)
    # adding a scheduler to reduce the learning_rate as soon as the validation loss stops decreasing,
    # this is to try to prevent overfitting of the model
    scheduler = ReduceLROnPlateau(optimizer, 'min')  # 'min' means reducing the LR when the metric stops decreasing

    train_model(config, model, optimizer, scheduler, loss_function, train_loader, val_loader, device)
    wandb.finish()

