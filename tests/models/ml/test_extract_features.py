"""Unit tests for extract_features() end-to-end pipeline."""

# ruff: noqa: N803, N806
import numpy as np
import pytest
from PIL import Image
from torchvision.datasets import ImageFolder

from jute_disease.models.ml import (
    HandcraftedFeatureExtractor,
    RawPixelFeatureExtractor,
    extract_features,
)
from jute_disease.utils import IMAGE_SIZE


@pytest.fixture
def mock_image_folder(tmp_path):
    """Build a minimal on-disk ImageFolder with 2 classes, 3 images each."""
    for cls in ["healthy", "diseased"]:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(3):
            img = Image.fromarray(
                np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            )
            img.save(cls_dir / f"img_{i}.jpg")
    return ImageFolder(root=str(tmp_path))


def test_raw_pixel_extract_features_shape(mock_image_folder):
    extractor = RawPixelFeatureExtractor(img_size=IMAGE_SIZE)
    X, y = extract_features(mock_image_folder, extractor)

    assert X.shape == (6, IMAGE_SIZE * IMAGE_SIZE * 3)
    assert y.shape == (6,)
    assert set(y.tolist()) == {0, 1}


@pytest.mark.slow
def test_handcrafted_extract_features_shape(mock_image_folder):
    extractor = HandcraftedFeatureExtractor()
    X, y = extract_features(mock_image_folder, extractor)

    assert X.shape[0] == 6
    assert X.shape[1] > 0
    assert X.dtype == np.float32
    assert y.shape == (6,)


def test_extract_features_label_alignment(mock_image_folder):
    """Labels must align with the class ordering from ImageFolder."""
    extractor = RawPixelFeatureExtractor(img_size=IMAGE_SIZE)
    X, y = extract_features(mock_image_folder, extractor)

    # ImageFolder sorts classes alphabetically: diseased=0, healthy=1
    assert set(y.tolist()) == {0, 1}
    assert len(y[y == 0]) == 3
    assert len(y[y == 1]) == 3
