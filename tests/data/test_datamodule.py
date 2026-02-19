"""Unit tests for the DataModule."""

import numpy as np
import pytest
from PIL import Image

from jute_disease.data import DataModule


@pytest.fixture
def mock_dataset_root(tmp_path):
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


def test_datamodule_setup_fit(mock_dataset_root):
    dm = DataModule(data_dir=mock_dataset_root, batch_size=2)
    dm.setup(stage="fit")

    assert len(dm.jute_train) == 10  # 2 classes * 5 images
    assert len(dm.jute_val) == 10
    assert dm.classes == ["diseased", "healthy"]


def test_datamodule_setup_test(mock_dataset_root):
    dm = DataModule(data_dir=mock_dataset_root, batch_size=2)
    dm.setup(stage="test")

    assert len(dm.jute_test) == 10
    assert dm.classes == ["diseased", "healthy"]


def test_datamodule_dataloaders(mock_dataset_root):
    dm = DataModule(data_dir=mock_dataset_root, batch_size=2)
    dm.setup()
    dm.setup(stage="test")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    assert len(next(iter(train_loader))[0]) == 2
    assert len(next(iter(val_loader))[0]) == 4  # batch_size * 2 for val
    assert len(next(iter(test_loader))[0]) == 4


def test_datamodule_kfold_setup(mock_dataset_root):
    # K-fold setup uses only the 'train' folder for both train and val
    dm = DataModule(data_dir=mock_dataset_root, batch_size=2, k_fold=5, fold_index=0)
    dm.setup(stage="fit")

    # 10 samples total, 5 folds -> 8 training, 2 validation
    assert len(dm.jute_train) == 8
    assert len(dm.jute_val) == 2


def test_datamodule_weighted_sampler(mock_dataset_root):
    # Imbalance: diseased=5, healthy=10 in train
    for i in range(5, 10):
        img = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
        img.save(mock_dataset_root / "train" / "healthy" / f"extra_{i}.jpg")

    dm = DataModule(data_dir=mock_dataset_root, batch_size=2, use_weighted_sampler=True)
    dm.setup(stage="fit")

    assert dm.sampler is not None
    assert len(dm.sampler) == 15  # 5 + 10
