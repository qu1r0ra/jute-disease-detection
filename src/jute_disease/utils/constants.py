from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve(strict=True).parents[3]
DATA_DIR = ROOT_DIR / "data"

# Image Configuration
IMAGE_SIZE = 1024
BATCH_SIZE = 32  # NOTE: Not final.
NUM_WORKERS = 4  # NOTE: Not final.

# Training Configuration
MAX_EPOCHS = 100  # NOTE: Not final.
LEARNING_RATE = 1e-3  # NOTE: Not final.
PATIENCE = 5

# Dataset Classes
CLASSES = [
    "Diehard",
    "Holed",
    "Mosaic",
    "Stem Soft Rot",
    "Fresh",
]
