import numpy as np
import torch
from PIL import Image

from jute_disease.config import IMAGE_SIZE
from jute_disease.data import PreprocessingSpec, classical_pipeline, create_pipeline


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


def test_pipelines_are_independent_and_seed_dependent():
    """No state is shared between pipelines; the seed alone decides augmentation."""
    img = Image.fromarray(
        np.random.default_rng(0).integers(0, 256, (400, 400, 3), np.uint8)
    )
    first = create_pipeline(size=64, is_train=True, is_dl=True, seed=1)(img)
    # Drawing from another pipeline in between must not disturb a fresh one.
    create_pipeline(size=64, is_train=True, is_dl=True, seed=2)(img)
    again = create_pipeline(size=64, is_train=True, is_dl=True, seed=1)(img)
    other = create_pipeline(size=64, is_train=True, is_dl=True, seed=2)(img)
    assert torch.equal(first, again)
    assert not torch.equal(first, other)


def test_eval_pipeline_is_deterministic_for_any_seed():
    img = Image.fromarray(
        np.random.default_rng(3).integers(0, 256, (300, 200, 3), np.uint8)
    )
    one = create_pipeline(size=64, is_train=False, is_dl=True, seed=1)(img)
    two = create_pipeline(size=64, is_train=False, is_dl=True, seed=99)(img)
    assert torch.equal(one, two)


def test_classical_pipeline_is_deterministic_and_unnormalised():
    img = Image.fromarray(
        np.random.default_rng(4).integers(0, 256, (300, 300, 3), np.uint8)
    )
    out = classical_pipeline(size=IMAGE_SIZE)(img)
    assert out.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out, classical_pipeline(size=IMAGE_SIZE)(img))


def test_preprocessing_spec_digest_changes_with_every_field():
    base = PreprocessingSpec()
    assert base.digest() == PreprocessingSpec().digest()
    for changed in (
        PreprocessingSpec(size=base.size + 1),
        PreprocessingSpec(cache_shortest_side=base.cache_shortest_side + 1),
        PreprocessingSpec(cache_format="webp"),
        PreprocessingSpec(resize="other"),
        PreprocessingSpec(interpolation="other"),
    ):
        assert changed.digest() != base.digest()
