"""Unit tests for TimmBackbone wrapper."""

from pathlib import Path

import pytest
import torch

from jute_disease.models.dl.backbone import TimmBackbone


@pytest.fixture
def backbone() -> TimmBackbone:
    return TimmBackbone(model_name="mobilenetv2_100", pretrained=False)


def test_timmbackbone_init(backbone: TimmBackbone) -> None:
    assert backbone.backbone is not None
    assert isinstance(backbone.out_features, int)
    assert backbone.out_features > 0


def test_timmbackbone_forward(backbone: TimmBackbone) -> None:
    x = torch.randn(2, 3, 224, 224)
    out = backbone(x)
    assert out.shape == (2, backbone.out_features)


def test_timmbackbone_no_checkpoint() -> None:
    """Instantiating without a checkpoint_path must not raise."""
    model = TimmBackbone(
        model_name="mobilenetv2_100", pretrained=False, checkpoint_path=None
    )
    assert model is not None


def test_timmbackbone_missing_checkpoint_raises(tmp_path: Path) -> None:
    """A non-existent checkpoint_path should raise an error from torch.load."""
    with pytest.raises(FileNotFoundError):
        TimmBackbone(
            model_name="mobilenetv2_100",
            pretrained=False,
            checkpoint_path=str(tmp_path / "nonexistent.ckpt"),
        )


def test_timmbackbone_custom_checkpoint_loading(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "fake.ckpt"
    # Create a fake checkpoint that matches the expected strings
    torch.save(
        {
            "state_dict": {
                "feature_extractor.backbone.some_random_layer.weight": torch.randn(1),
                "_orig_mod.feature_extractor.fc.weight": torch.randn(1),
                "other.weight": torch.randn(1),
            }
        },
        ckpt_path,
    )

    # Needs a real backbone layer to not crash if strict=False throws issues,
    # but strict=False is fine here.
    model = TimmBackbone(
        model_name="mobilenetv2_100", pretrained=False, checkpoint_path=str(ckpt_path)
    )
    assert model.out_features > 0


def test_timmbackbone_custom_checkpoint_no_matching_keys(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "fake_nomatch.ckpt"
    torch.save({"state_dict": {"random.weight": torch.randn(1)}}, ckpt_path)

    model = TimmBackbone(
        model_name="mobilenetv2_100", pretrained=False, checkpoint_path=str(ckpt_path)
    )
    assert model.out_features > 0


def test_backbone_override_features() -> None:
    """Ensure out_features can be manually overridden."""
    model = TimmBackbone(model_name="resnet18", pretrained=False, out_features=123)
    assert model.out_features == 123
