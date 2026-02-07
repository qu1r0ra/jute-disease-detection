import numpy as np
import pytest
from PIL import Image

from jute_disease.utils.constants import IMAGE_SIZE
from jute_disease.utils.feature_extractor import (
    HandcraftedFeatureExtractor,
    RawPixelFeatureExtractor,
)


@pytest.fixture
def sample_numpy_image() -> np.ndarray:
    """Create a dummy numpy image (H, W, C) for testing."""
    return np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Create a dummy PIL image for testing."""
    arr = np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_raw_pixel_extractor_numpy(sample_numpy_image):
    """Test RawPixelFeatureExtractor with numpy input."""
    extractor = RawPixelFeatureExtractor(img_size=IMAGE_SIZE)
    features = extractor.extract(sample_numpy_image)

    assert isinstance(features, np.ndarray)
    assert features.shape == (IMAGE_SIZE * IMAGE_SIZE * 3,)
    assert features.dtype == np.float32


def test_raw_pixel_extractor_pil(sample_pil_image):
    """Test RawPixelFeatureExtractor with PIL input."""
    extractor = RawPixelFeatureExtractor(img_size=IMAGE_SIZE)
    features = extractor.extract(sample_pil_image)

    assert isinstance(features, np.ndarray)
    assert features.shape == (IMAGE_SIZE * IMAGE_SIZE * 3,)


def test_handcrafted_extractor_numpy(sample_numpy_image):
    """Test HandcraftedFeatureExtractor with numpy input."""
    extractor = HandcraftedFeatureExtractor()
    features = extractor.extract(sample_numpy_image)

    assert isinstance(features, np.ndarray)
    assert features.shape[0] > 0
    assert features.dtype == np.float32


def test_handcrafted_extractor_pil(sample_pil_image):
    """Test HandcraftedFeatureExtractor with PIL input."""
    extractor = HandcraftedFeatureExtractor()
    features = extractor.extract(sample_pil_image)

    assert isinstance(features, np.ndarray)
    assert features.shape[0] > 0
