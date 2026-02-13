import logging

import matplotlib
import torch
from torchvision import datasets

from utils import (
    DEFAULT_TRANS,
    PATH_IN_TEST,
    PATH_OUT_MODEL,
    get_logger,
)


def main(logger: logging.Logger) -> None:
    logger.info(f"Retrieving trained neural network from {PATH_OUT_MODEL}...")
    neural_network = torch.load(PATH_OUT_MODEL, map_location='cpu', weights_only=False)
    neural_network.eval()

    logger.info(f'Retrieving testing data from "{PATH_IN_TEST}"...')
    testing_set = datasets.ImageFolder(PATH_IN_TEST, DEFAULT_TRANS)
    logger.info(f"Testing set size: {len(testing_set)}")

    logger.info("Preprocessing...")
    testing_set = torch.utils.data.DataLoader(testing_set, 8, True, num_workers=8)
    validation_data = torch.utils.data.DataLoader(
        validation_set, 16, True, num_workers=8
    )

    neural_network.fc = torch.nn.Linear(2048, 5)  #
    criterion = torch.nn.CrossEntropyLoss()  #
    optimiser = torch.optim.SGD(
        neural_network.parameters(), lr=1e-2, weight_decay=1e-4
    )  #

    trained_neural_network = train(
        training_data, validation_data, neural_network, criterion, optimiser, 5, logger
    )

    logger.info(f"Saving model to {PATH_OUT_MODEL}...")
    torch.save(trained_neural_network, PATH_OUT_MODEL)


if __name__ == "__main__":
    logger = get_logger(__name__)
    try:
        main(logger)
    except Exception as e:
        logger.exception(e)

