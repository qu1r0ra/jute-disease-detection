from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve(strict=True).parents[3]

DATA_DIR = ROOT_DIR / "data"
BY_CLASS_DIR = DATA_DIR / "by_class"
ML_SPLIT_DIR = DATA_DIR / "ml_split"
UNLABELED_DIR = DATA_DIR / "unlabeled"

ML_MODELS_DIR = ROOT_DIR / "artifacts" / "ml_models"
ML_FEATURES_DIR = ROOT_DIR / "artifacts" / "features"

# Data configurations
IMAGE_SIZE = 256
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}

# Execution defaults
BATCH_SIZE = 32
NUM_WORKERS = 4
SEEDS = [42, 1337, 7, 1234, 99]
DEFAULT_SEED = SEEDS[0]
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.01

# Task configurations
NUM_CLASSES = 6

# Weights & Biases
WANDB_ENTITY = "grade-descent"
WANDB_PROJECT = "jute-disease-detection"
