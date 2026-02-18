from jute_disease.utils import jute_data


def test_setup_jute_data_directories(tmp_path, monkeypatch):
    # Mock DATA_DIR and others in jute_data constants
    # Since jute_data imports from constants, we need to mock where it's used
    base_dir = tmp_path / "project"
    base_dir.mkdir()

    data_dir = base_dir / "data"
    data_dir.mkdir()

    by_class_dir = data_dir / "by_class"
    ml_split_dir = data_dir / "ml_split"

    # Create disease_classes.txt
    (data_dir / "disease_classes.txt").write_text("class1\nclass2\n")

    # Monkeypatch the constants used in jute_data
    monkeypatch.setattr(jute_data, "DATA_DIR", data_dir)
    monkeypatch.setattr(jute_data, "BY_CLASS_DIR", by_class_dir)
    monkeypatch.setattr(jute_data, "ML_SPLIT_DIR", ml_split_dir)

    jute_data.setup_jute_data_directory()

    assert (by_class_dir / "class1").exists()
    assert (by_class_dir / "class2").exists()
    assert (ml_split_dir / "train" / "class1").exists()
    assert (ml_split_dir / "val" / "class2").exists()


def test_split_jute_data(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    by_class_dir = data_dir / "by_class"
    by_class_dir.mkdir()

    ml_split_dir = data_dir / "ml_split"

    (by_class_dir / "class1").mkdir()
    # Create a dummy image
    (by_class_dir / "class1" / "img1.jpg").write_text("data")
    (by_class_dir / "class1" / "img2.jpg").write_text("data")
    (by_class_dir / "class1" / "img3.jpg").write_text("data")
    (by_class_dir / "class1" / "img4.jpg").write_text("data")

    monkeypatch.setattr(jute_data, "BY_CLASS_DIR", by_class_dir)
    monkeypatch.setattr(jute_data, "ML_SPLIT_DIR", ml_split_dir)
    monkeypatch.setattr(jute_data, "IMAGE_EXTENSIONS", [".jpg"])
    monkeypatch.setattr(jute_data, "SPLITS", {"train": 0.5, "val": 0.25, "test": 0.25})

    jute_data.split_jute_data()

    # 0.5 * 4 = 2 images in train
    assert len(list((ml_split_dir / "train" / "class1").glob("*.jpg"))) == 2
    # 0.25 * 4 = 1 image in val
    assert len(list((ml_split_dir / "val" / "class1").glob("*.jpg"))) == 1
    # 0.25 * 4 = 1 image in test
    assert len(list((ml_split_dir / "test" / "class1").glob("*.jpg"))) == 1
