"""Create a tiny synthetic jute dataset in data/by_class for pipeline smoke tests."""

import numpy as np
from PIL import Image

from jute_disease.config import BY_CLASS_DIR, DATA_DIR
from jute_disease.training.seed import seed_everything


def make_synthetic_data(per_class: int = 20, size: int = 64) -> None:
    """Write random RGB images for every class listed in disease_classes.txt."""
    seed_everything(42)
    rng = np.random.default_rng(42)
    classes = [
        line.strip()
        for line in (DATA_DIR / "disease_classes.txt").read_text().splitlines()
        if line.strip()
    ]
    for cls in classes:
        out_dir = BY_CLASS_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            pixels = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(out_dir / f"{cls}_{i:03d}.jpg")
