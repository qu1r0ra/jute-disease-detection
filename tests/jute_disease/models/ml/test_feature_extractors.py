import numpy as np
import pytest
from PIL import Image

from jute_disease.models.ml import CraftedFeatureExtractor, RawPixelFeatureExtractor
from jute_disease.utils import IMAGE_SIZE


@pytest.fixture
def sample_numpy_image() -> np.ndarray:
    """Create a dummy numpy image (H, W, C) for testing."""
    return np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Create a dummy PIL image for testing."""
    arr = np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_raw_pixel_extractor_numpy(sample_numpy_image: np.ndarray) -> None:
    """Test RawPixelFeatureExtractor with numpy input."""
    extractor = RawPixelFeatureExtractor(img_size=IMAGE_SIZE)
    features = extractor.extract(sample_numpy_image)

    assert isinstance(features, np.ndarray)
    assert features.shape == (IMAGE_SIZE * IMAGE_SIZE * 3,)
    assert features.dtype == np.float32


def test_raw_pixel_extractor_pil(sample_pil_image: Image.Image) -> None:
    """Test RawPixelFeatureExtractor with PIL input."""
    extractor = RawPixelFeatureExtractor(img_size=IMAGE_SIZE)
    features = extractor.extract(sample_pil_image)

    assert isinstance(features, np.ndarray)
    assert features.shape == (IMAGE_SIZE * IMAGE_SIZE * 3,)


def test_crafted_extractor_numpy(sample_numpy_image: np.ndarray) -> None:
    """Test CraftedFeatureExtractor with numpy input."""
    extractor = CraftedFeatureExtractor()
    features = extractor.extract(sample_numpy_image)

    assert isinstance(features, np.ndarray)
    assert features.shape[0] > 0
    assert features.dtype == np.float32


def test_crafted_extractor_pil(sample_pil_image: Image.Image) -> None:
    """Test CraftedFeatureExtractor with PIL input."""
    extractor = CraftedFeatureExtractor()
    features = extractor.extract(sample_pil_image)

    assert isinstance(features, np.ndarray)
    assert features.shape[0] > 0


def test_extract_features_logic(tmp_path) -> None:
    from unittest.mock import MagicMock

    import numpy as np

    from jute_disease.models.ml.features import extract_features

    # Mock dataset
    dataset = MagicMock()
    dataset.__len__.return_value = 2

    # Dummy image and label
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    label = 0

    def get_item_mock(idx: int):
        return (img, label)

    dataset.__getitem__.side_effect = get_item_mock

    extractor = RawPixelFeatureExtractor(img_size=32)

    # Test without cache
    x_feats, y = extract_features(dataset, extractor, cache_name=None)
    assert x_feats.shape == (2, 32 * 32 * 3)
    assert y.shape == (2,)

    # Test with cache saving
    import jute_disease.models.ml.features as features_module

    old_dir = features_module.ML_FEATURES_DIR
    features_module.ML_FEATURES_DIR = tmp_path

    try:
        x_feats2, y2 = extract_features(dataset, extractor, cache_name="test")
        assert x_feats2.shape == (2, 32 * 32 * 3)
        assert (tmp_path / "rawpixelfeatureextractor_test_X.npy").exists()

        # Test with cache loading
        dataset.__getitem__.side_effect = Exception("Should not be called")
        x_feats3, y3 = extract_features(dataset, extractor, cache_name="test")
        assert x_feats3.shape == (2, 32 * 32 * 3)
    finally:
        features_module.ML_FEATURES_DIR = old_dir
