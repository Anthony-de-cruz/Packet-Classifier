import logging

import torch
from matplotlib import pyplot
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torchvision import datasets

from utils import (
    DEFAULT_TRANS,
    PATH_IN_TEST,
    PATH_OUT_CM,
    PATH_OUT_MODEL,
    get_logger,
)


def test_and_plot_confusion_matrix(
    neural_network: torch.nn.Module,
    test_data: torch.utils.data.DataLoader,
    class_names: list[str],
    logger: logging.Logger,
) -> None:
    """"""
    logger.info("Running inference for confusion matrix...")

    all_preds = []
    all_labels = []

    neural_network.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_data):
            outputs = neural_network(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            print(f"Image: {i}/{len(test_data) - 1}", end="\r")
    logger.info("Inference complete.")

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(all_labels, all_preds, normalize="true"),
        display_labels=class_names,
    )

    figure, axes = pyplot.subplots(figsize=(8, 8))
    display.plot(ax=axes, colorbar=False, cmap="Blues", values_format=".0%")
    axes.set_title("Confusion Matrix", fontsize=16, pad=14)
    pyplot.tight_layout()

    logger.info(f"Saving confusion matrix to {PATH_OUT_CM}...")
    pyplot.savefig(PATH_OUT_CM, dpi=150)
    pyplot.show()
    pyplot.close(figure)


def main(logger: logging.Logger) -> None:
    logger.info(f"Retrieving trained neural network from {PATH_OUT_MODEL}...")
    neural_network = torch.load(PATH_OUT_MODEL, map_location="cpu", weights_only=False)

    logger.info(f'Retrieving testing data from "{PATH_IN_TEST}"...')
    testing_set = datasets.ImageFolder(PATH_IN_TEST, DEFAULT_TRANS)
    logger.info(f"Testing set size: {len(testing_set)}")

    testing_data = torch.utils.data.DataLoader(testing_set, 8, True, num_workers=8)

    test_and_plot_confusion_matrix(
        neural_network,
        testing_data,
        testing_set.classes,
        logger,
    )


if __name__ == "__main__":
    logger = get_logger(__name__)
    try:
        main(logger)
    except Exception as e:
        logger.exception(e)
