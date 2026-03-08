import numpy as np
import torch
from PIL import Image

from jute_disease.data import (
    dl_train_transforms,
    dl_val_transforms,
    ml_train_transforms,
    ml_val_transforms,
)
from jute_disease.data.transforms import create_pipeline
from jute_disease.utils import IMAGE_SIZE


def test_create_pipeline_ml():
    """ML pipeline should skip normalization and return numpy array."""
    pipeline = create_pipeline(size=256, is_train=False, is_dl=False)
    img = Image.fromarray(np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8))

    out = pipeline(img)
    assert isinstance(out, np.ndarray)
    assert out.shape == (256, 256, 3)
    # Check that it's not normalized (values around 0-255)
    assert out.max() > 1.0


def test_create_pipeline_dl():
    """DL pipeline should include normalization and return torch Tensor."""
    pipeline = create_pipeline(size=256, is_train=False, is_dl=True)
    img = Image.fromarray(np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8))

    out = pipeline(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 256, 256)
    # Normalized tensor usually has values around 0 (mean subtraction)
    assert out.min() < 0 or out.max() < 10.0


def test_create_pipeline_high_res():
    """Pipelines should respect the requested size."""
    pipeline = create_pipeline(size=512, is_train=False, is_dl=True)
    img = Image.fromarray(np.random.randint(0, 256, (600, 600, 3), dtype=np.uint8))

    out = pipeline(img)
    assert out.shape == (3, 512, 512)


def test_pipeline_seed_consistency():
    """Train pipeline with seed should be consistent for the same size."""
    pipeline1 = create_pipeline(size=256, is_train=True, is_dl=True)
    pipeline2 = create_pipeline(size=256, is_train=True, is_dl=True)

    img = Image.fromarray(np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8))

    out1 = pipeline1(img)
    out2 = pipeline2(img)

    assert torch.allclose(out1, out2)


# --- Global Instance Integration Tests ---


def test_global_ml_transforms():
    """Verify that global ML transforms are correctly instantiated."""
    img = Image.fromarray(np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8))

    train_out = ml_train_transforms(img)
    val_out = ml_val_transforms(img)

    assert train_out.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert val_out.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert isinstance(train_out, np.ndarray)


def test_global_dl_transforms():
    """Verify that global DL transforms are correctly instantiated."""
    img = Image.fromarray(np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8))

    train_out = dl_train_transforms(img)
    val_out = dl_val_transforms(img)

    assert train_out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert val_out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert isinstance(train_out, torch.Tensor)
