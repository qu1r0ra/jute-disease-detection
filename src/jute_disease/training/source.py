"""Training data setup for optional source-domain pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jute_disease.data import DataModule


def source_data_module(
    manifest_dir: Path | str,
    image_root: Path | str,
    target_manifest_dir: Path | str,
    batch_size: int,
    seed: int,
) -> DataModule:
    """Open source partitions only after pin, identity, terms and bytes verify."""
    from jute_disease.data import DataModule, load_source_manifest

    load_source_manifest(manifest_dir, image_root, target_manifest_dir)
    data_module = DataModule(
        manifest_dir=manifest_dir,
        image_root=image_root,
        batch_size=batch_size,
        seed=seed,
    )
    data_module.setup("fit")
    return data_module
