"""Unit tests for dataset download and preparation utilities."""

from pathlib import Path

import pytest

from jute_disease.data import stage_source
from jute_disease.data.download import package_version


def test_stage_source_preserves_official_layout(tmp_path: Path) -> None:
    """Official partitions are copied as published, never flattened or renamed."""
    raw_dir = tmp_path / "raw"
    target_dir = tmp_path / "target"

    for subset in ["train", "val"]:
        subset_dir = raw_dir / subset / "apple"
        subset_dir.mkdir(parents=True)
        (subset_dir / "img1.jpg").write_text(subset)
    (raw_dir / "notes.py").write_text("print('not an image')")

    stage_source(raw_dir, target_dir)

    assert (target_dir / "train" / "apple" / "img1.jpg").read_text() == "train"
    assert (target_dir / "val" / "apple" / "img1.jpg").read_text() == "val"
    assert not (target_dir / "notes.py").exists()


def test_package_version_is_read_from_the_download_path() -> None:
    path = "C:/cache/kagglehub/datasets/o/d/versions/7"
    assert package_version(path) == 7
    assert package_version(r"C:\cache\versions\12\train") == 12
    assert package_version("C:/somewhere/else") is None


def test_stage_source_skips_if_exists(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Test preparation skips if target directory already exists and is not empty."""
    import logging

    raw_dir = tmp_path / "raw"
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "exist").write_text("content")

    from jute_disease.data import download

    with caplog.at_level(logging.INFO):
        download.stage_source(raw_dir, target_dir)

    assert "already exists" in caplog.text


def test_stage_source_warns_without_images(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Test preparation logs warning if subset folder is missing."""
    import logging

    raw_dir = tmp_path / "raw"
    target_dir = tmp_path / "target"
    raw_dir.mkdir()

    from jute_disease.data import download

    with caplog.at_level(logging.WARNING):
        download.stage_source(raw_dir, target_dir)

    assert "No images found" in caplog.text
