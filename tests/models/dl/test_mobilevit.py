"""Unit tests for MobileViT backbone."""

import pytest
import torch

from jute_disease.models.dl import MobileViT


@pytest.fixture
def mobilevit():
    return MobileViT(pretrained=False)


def test_mobilevit_init(mobilevit):
    assert mobilevit.backbone is not None
    assert isinstance(mobilevit.out_features, int)
    assert mobilevit.out_features > 0


@pytest.mark.slow
def test_mobilevit_forward(mobilevit):
    x = torch.randn(2, 3, 256, 256)
    out = mobilevit(x)
    assert out.shape == (2, mobilevit.out_features)


def test_mobilevit_no_checkpoint():
    """Instantiating without a checkpoint_path must not raise."""
    model = MobileViT(pretrained=False, checkpoint_path=None)
    assert model is not None


@pytest.mark.slow
def test_mobilevit_missing_checkpoint_raises(tmp_path):
    """A non-existent checkpoint_path should raise an error from torch.load."""
    with pytest.raises(FileNotFoundError):
        MobileViT(pretrained=False, checkpoint_path=tmp_path / "nonexistent.ckpt")
