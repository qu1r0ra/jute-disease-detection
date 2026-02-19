"""Unit tests for data_utils: directory setup and train/val/test splitting."""

from jute_disease.utils import data_utils


def test_setup_data_directory(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    by_class_dir = data_dir / "by_class"
    ml_split_dir = data_dir / "ml_split"

    (data_dir / "disease_classes.txt").write_text("class1\nclass2\n")

    monkeypatch.setattr(data_utils, "DATA_DIR", data_dir)
    monkeypatch.setattr(data_utils, "BY_CLASS_DIR", by_class_dir)
    monkeypatch.setattr(data_utils, "ML_SPLIT_DIR", ml_split_dir)

    data_utils.setup_data_directory()

    assert (by_class_dir / "class1").exists()
    assert (by_class_dir / "class2").exists()
    assert (ml_split_dir / "train" / "class1").exists()
    assert (ml_split_dir / "val" / "class2").exists()


def test_setup_data_directory_missing_classes_file(tmp_path, monkeypatch, caplog):
    """Missing disease_classes.txt must log an error and not crash."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    monkeypatch.setattr(data_utils, "DATA_DIR", data_dir)
    monkeypatch.setattr(data_utils, "BY_CLASS_DIR", data_dir / "by_class")
    monkeypatch.setattr(data_utils, "ML_SPLIT_DIR", data_dir / "ml_split")

    data_utils.setup_data_directory()

    assert "not found" in caplog.text


def _make_split_fixture(tmp_path, monkeypatch, n_images=4):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    by_class_dir = data_dir / "by_class"
    by_class_dir.mkdir()
    ml_split_dir = data_dir / "ml_split"

    cls_dir = by_class_dir / "class1"
    cls_dir.mkdir()
    for i in range(n_images):
        (cls_dir / f"img{i}.jpg").write_text("data")

    monkeypatch.setattr(data_utils, "BY_CLASS_DIR", by_class_dir)
    monkeypatch.setattr(data_utils, "ML_SPLIT_DIR", ml_split_dir)
    monkeypatch.setattr(data_utils, "IMAGE_EXTENSIONS", [".jpg"])
    monkeypatch.setattr(data_utils, "SPLITS", {"train": 0.5, "val": 0.25, "test": 0.25})
    return ml_split_dir


def test_split_data(tmp_path, monkeypatch):
    ml_split_dir = _make_split_fixture(tmp_path, monkeypatch)

    data_utils.split_data()

    assert len(list((ml_split_dir / "train" / "class1").glob("*.jpg"))) == 2
    assert len(list((ml_split_dir / "val" / "class1").glob("*.jpg"))) == 1
    assert len(list((ml_split_dir / "test" / "class1").glob("*.jpg"))) == 1


def test_split_data_skips_if_exists(tmp_path, monkeypatch, caplog):
    """split_data must be a no-op if the split dir already has content."""
    import logging

    ml_split_dir = _make_split_fixture(tmp_path, monkeypatch)
    ml_split_dir.mkdir(parents=True)
    # Put a file *inside* the split dir so iterdir() returns non-empty
    (ml_split_dir / "sentinel").write_text("exists")

    with caplog.at_level(logging.INFO):
        data_utils.split_data()

    assert "Skipping" in caplog.text


def test_split_data_force_overwrites(tmp_path, monkeypatch):
    """force=True must delete and re-create the split directory."""
    ml_split_dir = _make_split_fixture(tmp_path, monkeypatch)

    data_utils.split_data()
    assert (ml_split_dir / "train" / "class1").exists()

    # Run again with force — should complete without error
    data_utils.split_data(force=True)
    assert len(list((ml_split_dir / "train" / "class1").glob("*.jpg"))) == 2
