"""Unit tests for the DataModule."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from jute_disease.data import DataModule


@pytest.fixture
def mock_dataset_root(tmp_path: Path) -> Path:
    """Create a mock data directory structure for testing DataModule."""
    data_dir = tmp_path / "data"
    for split in ["train", "val", "test"]:
        for cls in ["healthy", "diseased"]:
            cls_dir = data_dir / split / cls
            cls_dir.mkdir(parents=True)
            for i in range(5):
                img = Image.fromarray(
                    np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
                )
                img.save(cls_dir / f"img_{i}.jpg")
    return data_dir


def test_datamodule_setup_fit(mock_dataset_root: Path) -> None:
    dm = DataModule(data_dir=mock_dataset_root, batch_size=2)
    dm.setup(stage="fit")

    assert dm.jute_train is not None
    assert dm.jute_val is not None
    assert len(dm.jute_train) == 10  # 2 classes * 5 images
    assert len(dm.jute_val) == 10
    assert dm.classes == ["diseased", "healthy"]
    assert dm.num_classes == 2


def test_datamodule_setup_test(mock_dataset_root: Path) -> None:
    dm = DataModule(data_dir=mock_dataset_root, batch_size=2)
    dm.setup(stage="test")

    assert dm.jute_test is not None
    assert len(dm.jute_test) == 10
    assert dm.classes == ["diseased", "healthy"]


def test_datamodule_dataloaders(mock_dataset_root: Path) -> None:
    dm = DataModule(data_dir=mock_dataset_root, batch_size=2)
    dm.setup()
    dm.setup(stage="test")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    assert len(next(iter(train_loader))[0]) == 2
    assert len(next(iter(val_loader))[0]) == 4  # batch_size * 2 for val
    assert len(next(iter(test_loader))[0]) == 4


def test_datamodule_kfold_setup(mock_dataset_root: Path) -> None:
    # K-fold setup now merges 'train' and 'val' subfolders
    # Total samples = 10 (train) + 10 (val) = 20
    # 5 folds -> 16 training, 4 validation
    dm = DataModule(data_dir=mock_dataset_root, batch_size=2, k_fold=5, fold_index=0)
    dm.setup(stage="fit")

    assert dm.jute_train is not None
    assert dm.jute_val is not None
    assert len(dm.jute_train) == 16
    assert len(dm.jute_val) == 4
    assert dm.num_classes == 2


def test_datamodule_set_fold(mock_dataset_root: Path) -> None:
    """Test that set_fold switches the active subsets and indices."""
    dm = DataModule(data_dir=mock_dataset_root, k_fold=5, fold_index=0)
    dm.setup(stage="fit")

    assert hasattr(dm.jute_train, "indices")
    idx0 = dm.jute_train.indices.copy()  # type: ignore
    dm.set_fold(1)
    idx1 = dm.jute_train.indices.copy()  # type: ignore

    assert not np.array_equal(idx0, idx1)
    assert dm.hparams.fold_index == 1
    assert len(dm.jute_train) == 16  # type: ignore


def test_datamodule_weighted_sampler(mock_dataset_root: Path) -> None:
    # Imbalance: diseased=5, healthy=10 in train
    for i in range(5, 10):
        img = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
        img.save(mock_dataset_root / "train" / "healthy" / f"extra_{i}.jpg")

    dm = DataModule(data_dir=mock_dataset_root, batch_size=2, use_weighted_sampler=True)
    dm.setup(stage="fit")

    assert dm.sampler is not None
    assert len(dm.sampler) == 15  # 5 + 10


def test_datamodule_random_split(tmp_path: Path) -> None:
    """Test standard random splitting when subfolders are missing."""
    data_dir = tmp_path / "external_data"
    for cls in ["apple", "banana", "cherry"]:
        cls_dir = data_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(10):
            img = Image.fromarray(
                np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            )
            img.save(cls_dir / f"img_{i}.jpg")

    # 30 images total, 20% val -> 24 train, 6 val
    dm = DataModule(data_dir=data_dir, val_split=0.2, batch_size=2)
    dm.setup(stage="fit")

    assert dm.jute_train is not None
    assert dm.jute_val is not None
    assert dm.num_classes == 3
    assert len(dm.jute_train) == 24
    assert len(dm.jute_val) == 6
