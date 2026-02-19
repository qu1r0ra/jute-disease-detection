"""Jute dataset management — directory setup and train/val/test splitting."""

import argparse
import random
import shutil

from tqdm import tqdm

from jute_disease.utils.constants import (
    BY_CLASS_DIR,
    DATA_DIR,
    DEFAULT_SEED,
    IMAGE_EXTENSIONS,
    ML_SPLIT_DIR,
    SPLITS,
)
from jute_disease.utils.download import download_plant_doc, download_plant_village
from jute_disease.utils.logger import get_logger
from jute_disease.utils.seed import seed_everything

logger = get_logger(__name__)


def setup_data_directory():
    """Create the directory structure for the jute disease dataset."""
    disease_classes_file = DATA_DIR / "disease_classes.txt"

    if not disease_classes_file.exists():
        logger.error(f"Jute diseases class list not found at {disease_classes_file}")
        return

    with open(disease_classes_file) as f:
        classes = [line.strip() for line in f if line.strip()]

    logger.info("Initializing data directories...")

    for cls in classes:
        (BY_CLASS_DIR / cls).mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            (ML_SPLIT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    logger.info(f"Successfully created folders for {len(classes)} classes.")


def split_data(force: bool = False):
    """Split the jute dataset into train/val/test sets."""
    if ML_SPLIT_DIR.exists() and any(ML_SPLIT_DIR.iterdir()) and not force:
        logger.info(f"Split directory {ML_SPLIT_DIR} already exists. Skipping split.")
        return

    if force and ML_SPLIT_DIR.exists():
        logger.warning(f"Force flag set. Removing existing {ML_SPLIT_DIR}...")
        shutil.rmtree(ML_SPLIT_DIR)

    if not BY_CLASS_DIR.exists():
        logger.error(f"Source directory {BY_CLASS_DIR} does not exist.")
        return

    class_folders = [f for f in BY_CLASS_DIR.iterdir() if f.is_dir()]

    if not class_folders:
        logger.error(f"No class folders found in {BY_CLASS_DIR}")
        return

    logger.info(f"Found {len(class_folders)} classes. Starting split...")

    for class_folder in class_folders:
        class_name = class_folder.name

        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(list(class_folder.glob(f"*{ext}")))

        if not images:
            logger.warning(f"No images found for class '{class_name}'. Skipping.")
            continue

        random.shuffle(images)

        num_images = len(images)
        train_idx = int(num_images * SPLITS["train"])
        val_idx = train_idx + int(num_images * SPLITS["val"])

        assignments = {
            "train": images[:train_idx],
            "val": images[train_idx:val_idx],
            "test": images[val_idx:],
        }

        logger.info(f"Processing '{class_name}': {num_images} images")

        for split, split_images in assignments.items():
            split_path = ML_SPLIT_DIR / split / class_name
            split_path.mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(split_images, desc=f"  -> {split}", leave=False):
                shutil.copy2(img_path, split_path / img_path.name)

    logger.info("Data splitting complete!")


def initialize_data():
    """Run all data initialization tasks."""
    seed_everything(DEFAULT_SEED)
    setup_data_directory()
    split_data()
    download_plant_village()
    download_plant_doc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jute Data Management Utility")
    parser.add_argument(
        "command",
        nargs="?",
        default="init",
        choices=["setup", "split", "download", "init"],
        help="Command to run (default: init)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force split even if it already exists",
    )

    args = parser.parse_args()

    if args.command == "setup":
        setup_data_directory()
    elif args.command == "split":
        seed_everything(DEFAULT_SEED)
        split_data(force=args.force)
    elif args.command == "download":
        download_plant_village()
        download_plant_doc()
    elif args.command == "init":
        initialize_data()
