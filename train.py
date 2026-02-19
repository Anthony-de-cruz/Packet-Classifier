import datetime
import logging
import ssl

import torch
from torchvision import datasets, models

from utils import (
    DEFAULT_TRANS,
    PATH_IN_TRAINING,
    PATH_IN_VALID,
    PATH_OUT_MODEL,
    PATH_OUT_MODEL_ONNX,
    TRAINING_TRANS,
    get_logger,
)


def train(
    training_data: torch.utils.data.DataLoader,
    validation_data: torch.utils.data.DataLoader,
    neural_network: torch.nn.Module,
    criterion: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    num_epochs: int,
    logger: logging.Logger,
) -> torch.nn.Module:
    """Not currently GPU accelerated."""
    for epoch in range(num_epochs):
        logger.info(f"Training epoch {epoch}/{num_epochs - 1}...")
        training_start_time = datetime.datetime.now()
        training_accuracy = 0
        training_loss = 0
        neural_network = neural_network.train()

        for i, (image, label) in enumerate(training_data):
            image = torch.autograd.Variable(image)
            label = torch.autograd.Variable(label)

            out = neural_network(image)
            loss = criterion(out, label)

            optimiser.zero_grad()  # ?
            loss.backward()
            optimiser.step()

            # Accumulate accuracy/loss
            total = out.shape[0]
            _, pred_label = out.max(1)
            num_correct = (pred_label == label).sum().item()
            training_accuracy += num_correct / total
            training_loss += loss.item()

            print(f"Image: {i}/{len(training_data) - 1}", end="\r")

        training_time = datetime.datetime.now() - training_start_time
        valid_accuracy = 0
        valid_loss = 0
        neural_network = neural_network.eval()

        for image, label in validation_data:
            image = torch.autograd.Variable(image)
            label = torch.autograd.Variable(label)

            out = neural_network(image)
            loss = criterion(out, label)

            # Accumulate accuracy/loss
            total = out.shape[0]
            _, pred_label = out.max(1)
            num_correct = (pred_label == label).sum().item()
            valid_accuracy += num_correct / total
            valid_loss += loss.item()

        logger.info(f"Epoch {epoch}:")
        logger.info(f"  Training Time: {training_time}")
        logger.info(f"  Training Loss: {training_loss / len(training_data)}")
        logger.info(f"  Training Accuracy: {training_accuracy / len(training_data)}")
        logger.info(f"  Valid Loss: {valid_loss / len(training_data)}")
        logger.info(f"  Valid Accuracy: {valid_accuracy / len(training_data)}")

    return neural_network


def main(logger: logging.Logger) -> None:
    logger.info("Retrieving pretrained resnet50 neural network...")
    # Setup ssl cert in case new model weights need to be downloaded.
    ssl._create_default_https_context = ssl._create_unverified_context
    neural_network = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1, progress=True
    )

    logger.info(f'Retrieving training data from "{PATH_IN_TRAINING}"...')
    training_set = datasets.ImageFolder(PATH_IN_TRAINING, TRAINING_TRANS)
    logger.info(f"Training set size: {len(training_set)}")

    logger.info(f'Retrieving validation data from "{PATH_IN_VALID}"...')
    validation_set = datasets.ImageFolder(PATH_IN_VALID, DEFAULT_TRANS)
    logger.info(f"Validation set size: {len(validation_set)}")

    logger.info("Preprocessing...")
    training_data = torch.utils.data.DataLoader(training_set, 8, True, num_workers=8)
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

    logger.info(f"Saving Pytorch model to {PATH_OUT_MODEL}...")
    torch.save(trained_neural_network, PATH_OUT_MODEL)

    logger.info(f"Exporting ONNX model to {PATH_OUT_MODEL_ONNX}...")
    torch.onnx.export(
        trained_neural_network,
        (torch.randn(1, 3, 224, 224),),  # Dummy model input.
        PATH_OUT_MODEL_ONNX,
        input_names=["image"],
        output_names=["predictions"],
        dynamic_axes={"image": {0: "batch_size"}, "predictions": {0: "batch_size"}},
        opset_version=18,  #
    )


if __name__ == "__main__":
    logger = get_logger(__name__)
    try:
        main(logger)
    except Exception as e:
        logger.exception(e)
