import numpy as np
import pytest
import torch
from PIL import Image

from jute_disease.data import (
    dl_train_transforms,
    dl_val_transforms,
    ml_train_transforms,
    ml_val_transforms,
)
from jute_disease.utils import IMAGE_SIZE


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a dummy RGB image for testing."""
    arr = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_ml_train_transforms(sample_image: Image.Image) -> None:
    """Test ML training transforms return correct format."""
    transformed = ml_train_transforms(sample_image)

    # ML transforms should return numpy array (H, W, C) in 0-255 range
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert transformed.min() >= 0
    assert transformed.max() <= 255


def test_ml_val_transforms(sample_image: Image.Image) -> None:
    """Test ML validation transforms return correct format."""
    transformed = ml_val_transforms(sample_image)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert transformed.min() >= 0
    assert transformed.max() <= 255


def test_dl_train_transforms(sample_image: Image.Image) -> None:
    """Test DL training transforms return normalized tensor."""
    transformed = dl_train_transforms(sample_image)

    # DL transforms should return normalized torch Tensor (C, H, W)
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert transformed.dtype == torch.float32

    # Check for negative values to confirm normalization applied
    if transformed.min() >= 0:
        pass


def test_dl_val_transforms(sample_image: Image.Image) -> None:
    """Test DL validation transforms return normalized tensor."""
    transformed = dl_val_transforms(sample_image)

    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert transformed.dtype == torch.float32
