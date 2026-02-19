"""Download and prepare external datasets for pre-training."""

import shutil
from pathlib import Path

import kagglehub
from tqdm import tqdm

from jute_disease.utils.constants import DATA_DIR
from jute_disease.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_dataset_subsets(
    raw_dir: Path, target_dir: Path, subsets: list[str]
) -> None:
    """Consolidate dataset subsets into a single by_class directory."""
    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info(f"Target directory {target_dir} already exists. Skipping prep.")
        return

    logger.info(f"Preparing dataset in {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)

    for subset in subsets:
        subset_path = raw_dir / subset
        if not subset_path.exists():
            logger.warning(f"Subset folder {subset_path} not found. Skipping.")
            continue

        class_folders = [d for d in subset_path.iterdir() if d.is_dir()]

        for class_folder in tqdm(class_folders, desc=f"Processing {subset}"):
            class_name = class_folder.name
            dest_class_dir = target_dir / class_name
            dest_class_dir.mkdir(exist_ok=True)

            for img_file in class_folder.iterdir():
                if img_file.is_file():
                    # Prefix with subset to avoid name collisions
                    dest_file = dest_class_dir / f"{subset}_{img_file.name}"
                    shutil.copy2(img_file, dest_file)

    logger.info(f"Preparation complete for {target_dir.name}!")


def download_and_prepare_kaggle_data(
    dataset_name: str, kaggle_id: str, target_dirname: str, subsets: list[str]
) -> None:
    """Generic download and prepare function for Kaggle data."""
    target_dir = DATA_DIR / "external" / target_dirname

    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info(f"{dataset_name} seems ready. Skipping.")
        return

    logger.info(f"Downloading {dataset_name}...")
    path = kagglehub.dataset_download(kaggle_id)
    downloaded_dir = Path(path)
    logger.info(f"Downloaded to {downloaded_dir}")

    prepare_dataset_subsets(downloaded_dir, target_dir, subsets)


def download_plant_village() -> None:
    """Download the Plant Village dataset and prepare it."""
    download_and_prepare_kaggle_data(
        "PlantVillage",
        "mohitsingh1804/plantvillage",
        "plant_village",
        ["train", "val"],
    )


def download_plant_doc() -> None:
    """Download the PlantDoc dataset and prepare it."""
    download_and_prepare_kaggle_data(
        "PlantDoc",
        "nirmalsankalana/plantdoc-dataset",
        "plantdoc",
        ["train", "test"],
    )
