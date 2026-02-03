import random
import shutil

from tqdm import tqdm

from jute_disease_pest.utils.constants import (
    BY_CLASS_DIR,
    DEFAULT_SEED,
    IMAGE_EXTENSIONS,
    ML_SPLIT_DIR,
    SPLITS,
)
from jute_disease_pest.utils.logger import get_logger
from jute_disease_pest.utils.seed import seed_everything

logger = get_logger(__name__)


def split_dataset(force: bool = False):
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


if __name__ == "__main__":
    seed_everything(DEFAULT_SEED)
    split_dataset()
