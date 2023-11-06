import argparse
import os

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, random_split

from PositionToEvaluationDataset import PositionToEvaluationDataset
from src.model.ChessmaitMlp1 import ChessmaitMlp1

############################################################
# This is the central Python script to perform the training
############################################################

POSITION_TO_EVALUATION_FILE_PATH = os.path.join("data", "preprocessed", "kaggle")


def check_cuda():
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
    _config.train_percentage = 0.7  # percentage of data which is used for training
    _config.val_percentage = 0.15  # percentage of data which is used for validation (rest is for testing)
    _config.learning_rate = 0.001
    _config.betas = (0.90, 0.99)  # needed for Adam optimizer
    _config.eps = 1e-8  # needed for Adam optimizer
    _config.epochs = 15
    _config.batch_size = 64

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
    _dataset = PositionToEvaluationDataset(os.path.join(POSITION_TO_EVALUATION_FILE_PATH, "kaggle_preprocessed.csv"))
    train_size = int(_config.train_percentage * len(_dataset))
    val_size = int(_config.val_percentage * len(_dataset))
    test_size = len(_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(_dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    _train_loader = DataLoader(train_dataset, batch_size=_config.batch_size, shuffle=True)
    _val_loader = DataLoader(val_dataset, batch_size=_config.batch_size, shuffle=False)
    _test_loader = DataLoader(test_dataset, batch_size=_config.batch_size, shuffle=False)

    return _train_loader, _val_loader, _test_loader


def train_model(_config: argparse.Namespace,
                _model: nn.Module,
                _optimizer: torch.optim.Adam,
                _loss_function: nn.MSELoss,
                _train_loader: DataLoader,
                _val_loader: DataLoader):
    wandb.watch(_model)

    best_val_loss = float('inf')

    for epoch in range(1, _config.epochs + 1):
        # Training
        _model.train()
        train_loss = 0.0

        for position, evaluation in _train_loader:
            _optimizer.zero_grad()
            predicted_evaluation = _model(position)

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
                loss = _loss_function(predicted_evaluation, evaluation)
                val_loss += loss.item()

        wandb.log({"epoch": epoch,
                   "training loss": train_loss,
                   "validation loss": val_loss
                   })


if __name__ == "__main__":
    print("Starting training process ...")

    device = check_cuda()
    config = get_training_configuration()
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Create instances of the model, the optimizer and the loss function
    model = ChessmaitMlp1()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.betas, eps=config.eps)
    loss_function = nn.MSELoss()

    wandb.init(project="chessmait", config=vars(config))
    train_model(config, model, optimizer, loss_function, train_loader, val_loader)
