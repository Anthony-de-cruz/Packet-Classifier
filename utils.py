import logging
import os
import sys
from torchvision import transforms

PATH_IN = os.path.join("data")
PATH_OUT = os.path.join("out")

PATH_IN_TRAINING = os.path.join(PATH_IN, "training")
PATH_IN_TEST = os.path.join(PATH_IN, "test")
PATH_IN_VALID = os.path.join(PATH_IN, "valid")

PATH_OUT_MODEL = os.path.join(PATH_OUT, "model.pth")
PATH_OUT_MODEL_ONNX = os.path.join(PATH_OUT, "model.onnx")
PATH_OUT_CM = os.path.join(PATH_OUT, "confusion_matrix.png")
PATH_OUT_LOG = os.path.join(PATH_OUT, "log.txt")

# Current value assumed for a CPU.
# Increase to at least 16(?) on a GPU.
BATCH_SIZE = 1

# Adjust for CPU core count.
WORKER_COUNT = 4

# Transformations for a training dataset.
TRAINING_TRANS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            [
                0.5,
            ],
            [
                0.5,
            ],
        ),
    ]
)

# Transformations for an input dataset.
DEFAULT_TRANS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [
                0.5,
            ],
            [
                0.5,
            ],
        ),
    ]
)


def get_logger(name: str) -> logging.Logger:
    """Setup logging."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    if not os.path.isdir(PATH_OUT):
        os.mkdir(PATH_OUT)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    file_handler = logging.FileHandler(filename=PATH_OUT_LOG)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)

    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger
