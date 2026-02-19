"""Unit tests for dataset download and preparation utilities."""

from jute_disease.utils.download import prepare_dataset_subsets


def test_prepare_dataset_subsets(tmp_path):
    """Test consolidating subsets into by_class directory."""
    raw_dir = tmp_path / "raw"
    target_dir = tmp_path / "target"

    # Create mock raw structure: raw/train/apple/img1.jpg, raw/val/apple/img2.jpg
    for subset in ["train", "val"]:
        subset_dir = raw_dir / subset / "apple"
        subset_dir.mkdir(parents=True)
        (subset_dir / "img1.jpg").write_text("data")

    prepare_dataset_subsets(raw_dir, target_dir, ["train", "val"])

    # Should result in target/apple/train_img1.jpg and target/apple/val_img1.jpg
    assert (target_dir / "apple" / "train_img1.jpg").exists()
    assert (target_dir / "apple" / "val_img1.jpg").exists()


def test_prepare_dataset_subsets_skips_if_exists(tmp_path, caplog):
    """Test preparation skips if target directory already exists and is not empty."""
    import logging

    raw_dir = tmp_path / "raw"
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "exist").write_text("content")

    from jute_disease.utils import download

    with caplog.at_level(logging.INFO):
        download.prepare_dataset_subsets(raw_dir, target_dir, ["train"])

    assert "already exists" in caplog.text


def test_prepare_dataset_subsets_missing_folder(tmp_path, caplog):
    """Test preparation logs warning if subset folder is missing."""
    import logging

    raw_dir = tmp_path / "raw"
    target_dir = tmp_path / "target"
    raw_dir.mkdir()

    from jute_disease.utils import download

    with caplog.at_level(logging.WARNING):
        download.prepare_dataset_subsets(raw_dir, target_dir, ["missing"])

    assert "not found" in caplog.text
