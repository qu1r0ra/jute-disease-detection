import re
import shutil
from pathlib import Path

import kagglehub
from tqdm import tqdm

from jute_disease.config import DATA_DIR, IMAGE_EXTENSIONS
from jute_disease.data.sources import PLANTDOC, PLANTVILLAGE, SourceSpec
from jute_disease.training.logger import get_logger

logger = get_logger(__name__)

_VERSION_PATTERN = re.compile(r"versions[/\\](\d+)")
_EXTENSIONS = {ext.lower() for ext in IMAGE_EXTENSIONS}


def stage_source(raw_dir: Path, target_dir: Path) -> None:
    """Copy a downloaded package's images, keeping the publisher's folder layout.

    Official partitions are preserved exactly: nothing is flattened, renamed or
    merged, so provenance and the publisher's own split stay checkable.
    """
    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info(f"Target directory {target_dir} already exists. Skipping prep.")
        return

    images = sorted(
        p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in _EXTENSIONS
    )
    if not images:
        logger.warning(f"No images found in {raw_dir}. Skipping.")
        return

    logger.info(f"Staging dataset in {target_dir}...")
    for image in tqdm(images, desc="Staging images"):
        destination = target_dir / image.relative_to(raw_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, destination)
    logger.info(f"Staging complete for {target_dir.name}!")


def package_version(path: Path | str) -> int | None:
    """Return the Kaggle package version encoded in a download path, if any."""
    match = _VERSION_PATTERN.search(str(path))
    return int(match.group(1)) if match else None


def download_source(spec: SourceSpec) -> tuple[Path, int | None]:
    """Download (or reuse) a pinned source package into ``data/external``.

    Returns the staged folder and the package version recorded from the
    download path (``None`` when the data was already staged earlier).
    """
    target_dir = DATA_DIR / "external" / spec.name.lower()
    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info(f"{spec.name} seems ready. Skipping.")
        return target_dir, None

    handle = spec.kaggle_id
    if spec.version is not None:
        handle = f"{handle}/versions/{spec.version}"
    logger.info(f"Downloading {handle}...")
    downloaded = Path(kagglehub.dataset_download(handle))
    logger.info(f"Downloaded to {downloaded}")

    stage_source(downloaded, target_dir)
    return target_dir, package_version(downloaded)


def download_plant_village() -> None:
    download_source(PLANTVILLAGE)


def download_plant_doc() -> None:
    download_source(PLANTDOC)
