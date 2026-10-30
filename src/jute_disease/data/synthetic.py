"""Structured synthetic dataset for tests and pipeline smoke runs.

Unlike random noise, these images have a stable perceptual hash and a planted
leakage structure, so the identity and split logic can be exercised end to end:

* ``rotated copies``: a same-label image repeated at a quarter turn (a
  perceptual candidate under the dihedral hash),
* ``look-alike``: a lightly altered copy of one class's image filed under another
  class (an unresolved cross-label candidate),
* ``byte copy``: a byte-identical duplicate under a new name,
* ``quarantine files``: Healthy files named like the quarantined contribution.

The layout is ``<root>/<label>/<file>``; nothing is read from or written to real data.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from jute_disease.config import CLASS_NAMES, DEFAULT_SEED

QUARANTINE_PREFIX = "IMG_20240502"


@dataclass
class SyntheticPlan:
    """Where the planted structure lives, as paths relative to the root."""

    root: Path
    files: list[str] = field(default_factory=list)
    rotated_pairs: list[tuple[str, str]] = field(default_factory=list)
    lookalike_pairs: list[tuple[str, str]] = field(default_factory=list)
    byte_copies: list[tuple[str, str]] = field(default_factory=list)
    quarantine_files: list[str] = field(default_factory=list)


def _structured_image(rng: np.random.Generator, size: int) -> Image.Image:
    """A smooth random field: distinct per image, stable under resizing."""
    coarse = rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)
    return Image.fromarray(coarse).resize((size, size), Image.Resampling.BICUBIC)


def make_synthetic_data(
    root: Path,
    per_class: int = 20,
    size: int = 64,
    seed: int = DEFAULT_SEED,
    labels: tuple[str, ...] = CLASS_NAMES,
    rotated_per_class: int = 2,
    quarantine_count: int = 0,
) -> SyntheticPlan:
    """Write a structured dataset under ``root`` and return the planted plan.

    Raises:
        ValueError: ``per_class`` cannot hold the planted copies, or ``root`` is
            not empty (existing files are never overwritten).
    """
    root = Path(root)
    if per_class <= rotated_per_class:
        raise ValueError("per_class must exceed rotated_per_class")
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Refusing to write into a non-empty directory: {root}")
    rng = np.random.default_rng(seed)
    plan = SyntheticPlan(root=root)

    def save(label: str, name: str, image: Image.Image) -> str:
        path = root / label / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        relative = path.relative_to(root).as_posix()
        plan.files.append(relative)
        return relative

    originals: dict[str, list[tuple[str, Image.Image]]] = {}
    for label in labels:
        slug = label.replace(" ", "_")
        made = []
        for index in range(per_class - rotated_per_class):
            image = _structured_image(rng, size)
            made.append((save(label, f"{slug}_{index:03d}.png", image), image))
        for index in range(rotated_per_class):
            source_path, source = made[index]
            turned = source.transpose(Image.Transpose.ROTATE_90)
            copy = save(label, f"{slug}_rot_{index:03d}.png", turned)
            plan.rotated_pairs.append((source_path, copy))
        originals[label] = made

    if len(labels) > 1:
        first, second = labels[0], labels[1]
        source_path, source = originals[first][-1]
        shifted = Image.fromarray(
            np.clip(np.asarray(source, dtype=np.int16) + 2, 0, 255).astype(np.uint8)
        )
        copy = save(second, f"{second.replace(' ', '_')}_lookalike.png", shifted)
        plan.lookalike_pairs.append((source_path, copy))

    source_path = originals[labels[0]][-2][0]
    duplicate = root / labels[0] / f"{labels[0].replace(' ', '_')}_bytecopy.png"
    duplicate.write_bytes((root / source_path).read_bytes())
    plan.files.append(duplicate.relative_to(root).as_posix())
    plan.byte_copies.append((source_path, plan.files[-1]))

    for index in range(quarantine_count):
        label = "Healthy" if "Healthy" in labels else labels[0]
        name = save(
            label, f"{QUARANTINE_PREFIX}_{index:04d}.jpg", _structured_image(rng, size)
        )
        plan.quarantine_files.append(name)
    plan.files.sort()
    return plan
