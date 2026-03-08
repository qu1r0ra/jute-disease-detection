import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from jute_disease.utils import get_logger

logger = get_logger(__name__)


def get_image_size(img_path):
    try:
        with Image.open(img_path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def main():
    image_dir = Path("data/by_class")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                image_paths.append(Path(root) / file)

    logger.info(f"Total images found: {len(image_paths)}")

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            tqdm(
                executor.map(get_image_size, image_paths),
                total=len(image_paths),
                desc="Processing images",
            )
        )

    # Filter out failed reads
    sizes = [s for s in results if s is not None]

    if not sizes:
        logger.warning("No valid images processed.")
        return

    df = pd.DataFrame(sizes, columns=["Width", "Height"])

    logger.info("\n--- Image Dimension Statistics ---")
    logger.info(f"\n{df.describe()}")

    logger.info("\n--- Most Common Resolutions ---")
    logger.info(f"\n{df.value_counts().head(10)}")

    # Check if we can go higher (e.g. 384, 512)
    resolutions_to_check = [256, 384, 512, 1024]
    logger.info("\n--- Feasibility Check ---")
    for res in resolutions_to_check:
        count = len(df[(df["Width"] >= res) & (df["Height"] >= res)])
        percentage = (count / len(df)) * 100
        logger.info(f"Images >= {res}x{res}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
