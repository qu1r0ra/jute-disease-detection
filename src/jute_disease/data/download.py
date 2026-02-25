"""Download and prepare external datasets for pre-training."""

import shutil
from pathlib import Path

import kagglehub
from tqdm import tqdm

from jute_disease.utils.constants import DATA_DIR, IMAGE_EXTENSIONS
from jute_disease.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_dataset_subsets(raw_dir: Path, target_dir: Path) -> None:
    """Consolidate dataset subsets into a single directory by class."""
    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info(f"Target directory {target_dir} already exists. Skipping prep.")
        return

    logger.info(f"Preparing dataset in {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(raw_dir.rglob(f"*{ext}"))

    if not images:
        logger.warning(f"No images found in {raw_dir}. Skipping.")
        return

    for img_file in tqdm(images, desc="Consolidating classes"):
        class_name = img_file.parent.name
        dest_class_dir = target_dir / class_name
        dest_class_dir.mkdir(exist_ok=True)

        dest_file = dest_class_dir / f"{img_file.parent.parent.name}_{img_file.name}"
        shutil.copy2(img_file, dest_file)

    logger.info(f"Preparation complete for {target_dir.name}!")


def download_and_prepare_kaggle_data(dataset_name: str, kaggle_id: str) -> None:
    """Generic download and prepare function for Kaggle data."""
    target_dirname = dataset_name.lower()
    target_dir = DATA_DIR / "external" / target_dirname

    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info(f"{dataset_name} seems ready. Skipping.")
        return

    logger.info(f"Downloading {dataset_name}...")
    path = kagglehub.dataset_download(kaggle_id)
    downloaded_dir = Path(path)
    logger.info(f"Downloaded to {downloaded_dir}")

    prepare_dataset_subsets(downloaded_dir, target_dir)


def download_plant_village() -> None:
    """Download the Plant Village dataset and prepare it."""
    download_and_prepare_kaggle_data(
        "PlantVillage",
        "mohitsingh1804/plantvillage",
    )


def download_plant_doc() -> None:
    """Download the PlantDoc dataset and prepare it."""
    download_and_prepare_kaggle_data(
        "PlantDoc",
        "nirmalsankalana/plantdoc-dataset",
    )
