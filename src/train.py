import argparse
import os
from typing import Union

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.PositionToEvaluationDataset import PositionToEvaluationDataset
from src.model.ChessmaitMlp2 import ChessmaitMlp2

############################################################
# This is the central Python script to perform the training
############################################################

PATH_TO_DATAFILE = os.path.join("data", "preprocessed", "kaggle")

############################################################################
# Make sure these parameters are correctly set before you start the training
############################################################################
DATA_FILE = "kaggle_preprocessed.csv"
NUMBER_OF_GAMES_FOR_TRAINING = 49946
WANDB_REPORTING = True
REGRESSION_TRAINING = True

model = ChessmaitMlp2()


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
    _config.number_of_games_for_training = NUMBER_OF_GAMES_FOR_TRAINING
    _config.train_percentage = 0.7  # percentage of data which is used for training
    _config.val_percentage = 0.15  # percentage of data which is used for validation (rest is for testing)
    _config.learning_rate = 0.001
    _config.betas = (0.90, 0.99)  # needed for Adam optimizer
    _config.eps = 1e-8  # needed for Adam optimizer
    _config.epochs = 15
    _config.batch_size = 1024

    return _config


def get_dataloaders(_config: argparse.Namespace, _device: str) -> (DataLoader, DataLoader, DataLoader):
    """

    Parameters
    ----------
    _config : argparse.Namespace
        configuration values for this training
    _device : str
        device on which the tensors are processed

    Returns
    -------
    tuple
        Dataloader for training, validation and testing

    """
    print("Prepare dataloaders ...")
    dataset = PositionToEvaluationDataset(os.path.join(PATH_TO_DATAFILE, DATA_FILE), _device)

    torch.manual_seed(42)

    train_size = int(_config.train_percentage * len(dataset))
    val_size = int(_config.val_percentage * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    _train_loader = DataLoader(train_dataset, batch_size=_config.batch_size, shuffle=True)
    _val_loader = DataLoader(val_dataset, batch_size=_config.batch_size, shuffle=False)
    _test_loader = DataLoader(test_dataset, batch_size=_config.batch_size, shuffle=False)

    print("Prepare dataloaders finished ...")
    return _train_loader, _val_loader, _test_loader


def train_model(_config: argparse.Namespace,
                _model: nn.Module,
                _optimizer: torch.optim.Adam,
                _scheduler,
                _loss_function: Union[nn.CrossEntropyLoss, nn.MSELoss],
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
            predicted_evaluation = _model(position)

            if REGRESSION_TRAINING:
                evaluation = evaluation.unsqueeze(1)  # Reshapes [64] to [64, 1] to match the predicted_evaluation
            loss = _loss_function(predicted_evaluation, evaluation)
            loss.backward()
            _optimizer.step()
            train_loss += loss.item()

        # Validation
        _model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for position, evaluation in _val_loader:
                predicted_evaluation = _model(position)

                if REGRESSION_TRAINING:
                    evaluation = evaluation.unsqueeze(1)  # Reshapes [64] to [64, 1] to match the predicted_evaluation
                loss = _loss_function(predicted_evaluation, evaluation)
                val_loss += loss.item()

        epoch_result = {"epoch": epoch,
                        "training loss": train_loss,
                        "validation loss": val_loss
                        }
        print(epoch_result)
        if WANDB_REPORTING:
            wandb.log(epoch_result)

        # step the scheduler - adjust the learning rate if validation loss stops decreasing
        _scheduler.step(val_loss)

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

    device = get_device()
    config = get_training_configuration()
    train_loader, val_loader, test_loader = get_dataloaders(config, device)

    if WANDB_REPORTING:
        config.model = type(model).__name__
        wandb.init(project="chessmait", config=vars(config))

    if REGRESSION_TRAINING:
        loss_function = nn.MSELoss()
    else:
        loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.betas, eps=config.eps)
    # adding a scheduler to reduce the learning_rate as soon as the validation loss stops decreasing,
    # this is to try to prevent overfitting of the model
    scheduler = ReduceLROnPlateau(optimizer, 'min')  # 'min' means reducing the LR when the metric stops decreasing

    train_model(config, model, optimizer, scheduler, loss_function, train_loader, val_loader, device)
    wandb.finish()
