"""Unit tests for MobileViT backbone."""

from pathlib import Path

import pytest
import torch

from jute_disease.models.dl import MobileViT


@pytest.fixture
def mobilevit() -> MobileViT:
    return MobileViT(pretrained=False)


def test_mobilevit_init(mobilevit: MobileViT) -> None:
    assert mobilevit.backbone is not None
    assert isinstance(mobilevit.out_features, int)
    assert mobilevit.out_features > 0


def test_mobilevit_forward(mobilevit: MobileViT) -> None:
    x = torch.randn(2, 3, 256, 256)
    out = mobilevit(x)
    assert out.shape == (2, mobilevit.out_features)


def test_mobilevit_no_checkpoint() -> None:
    """Instantiating without a checkpoint_path must not raise."""
    model = MobileViT(pretrained=False, checkpoint_path=None)
    assert model is not None


def test_mobilevit_missing_checkpoint_raises(tmp_path: Path) -> None:
    """A non-existent checkpoint_path should raise an error from torch.load."""
    with pytest.raises(FileNotFoundError):
        MobileViT(pretrained=False, checkpoint_path=tmp_path / "nonexistent.ckpt")


def test_mobilevit_custom_checkpoint_loading(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "fake.ckpt"
    # Create a fake checkpoint that matches the expected strings
    torch.save(
        {
            "state_dict": {
                "feature_extractor.backbone.conv1.weight": torch.randn(1),
                "feature_extractor.fc.weight": torch.randn(1),
                "other.weight": torch.randn(1),
            }
        },
        ckpt_path,
    )

    # Needs a real backbone layer to not crash if strict=False throws issues,
    # but strict=False is fine here.
    model = MobileViT(pretrained=False, checkpoint_path=ckpt_path)
    assert model.out_features > 0


def test_mobilevit_custom_checkpoint_no_matching_keys(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "fake_nomatch.ckpt"
    torch.save({"state_dict": {"random.weight": torch.randn(1)}}, ckpt_path)

    model = MobileViT(pretrained=False, checkpoint_path=ckpt_path)
    assert model.out_features > 0
