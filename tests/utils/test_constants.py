"""Unit tests for constants: types, values, and path consistency."""

from pathlib import Path

from jute_disease.utils import (
    BATCH_SIZE,
    BY_CLASS_DIR,
    DATA_DIR,
    DEFAULT_SEED,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    ML_MODELS_DIR,
    ML_SPLIT_DIR,
    NUM_WORKERS,
    ROOT_DIR,
    SEEDS,
    SPLITS,
    UNLABELED_DIR,
    WANDB_ENTITY,
    WANDB_PROJECT,
)


def test_paths_are_path_objects():
    for p in [
        ROOT_DIR,
        DATA_DIR,
        ML_MODELS_DIR,
        BY_CLASS_DIR,
        ML_SPLIT_DIR,
        UNLABELED_DIR,
    ]:
        assert isinstance(p, Path), f"{p!r} is not a Path"


def test_root_dir_is_project_root():
    """ROOT_DIR must be the repo root (contains pyproject.toml)."""
    assert (ROOT_DIR / "pyproject.toml").exists()


def test_paths_are_nested_under_root():
    assert str(DATA_DIR).startswith(str(ROOT_DIR))
    assert str(ML_MODELS_DIR).startswith(str(ROOT_DIR))


def test_splits_sum_to_one():
    assert abs(sum(SPLITS.values()) - 1.0) < 1e-9


def test_splits_has_required_keys():
    assert set(SPLITS.keys()) == {"train", "val", "test"}


def test_numeric_constants_are_positive():
    assert IMAGE_SIZE > 0
    assert BATCH_SIZE > 0
    assert NUM_WORKERS >= 0


def test_seeds_contains_default():
    assert DEFAULT_SEED in SEEDS
    assert len(SEEDS) > 0


def test_image_extensions_are_strings():
    assert len(IMAGE_EXTENSIONS) > 0
    for ext in IMAGE_EXTENSIONS:
        assert ext.startswith("."), f"Extension {ext!r} must start with '.'"


def test_wandb_strings_are_nonempty():
    assert isinstance(WANDB_ENTITY, str) and WANDB_ENTITY
    assert isinstance(WANDB_PROJECT, str) and WANDB_PROJECT
