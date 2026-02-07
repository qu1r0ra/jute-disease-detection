from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve(strict=True).parents[3]
DATA_DIR = ROOT_DIR / "data"
ML_MODELS_DIR = ROOT_DIR / "dump" / "ml_models"
BY_CLASS_DIR = DATA_DIR / "by_class"
ML_SPLIT_DIR = DATA_DIR / "ml_split"
UNLABELED_DIR = DATA_DIR / "unlabeled"

# Image Configuration
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
IMAGE_SIZE = 256
BATCH_SIZE = 256
NUM_WORKERS = 4

# Training Configuration
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
PATIENCE = 5
SEEDS = [42, 1337, 7, 1234, 99]
DEFAULT_SEED = SEEDS[0]

# Dataset Configuration
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}
