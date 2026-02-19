"""Unit tests for the DL Classifier (LightningModule)."""

import pytest
import torch

from jute_disease.models.dl import Classifier, TimmBackbone


@pytest.fixture
def backbone():
    return TimmBackbone(model_name="resnet18", pretrained=False)


@pytest.fixture
def classifier(backbone):
    return Classifier(
        feature_extractor=backbone,
        num_classes=6,
        lr=1e-3,
        freeze_backbone=False,
        compile_model=False,
    )


def test_classifier_requires_out_features():
    """Classifier must reject a feature extractor without out_features."""
    import torch.nn as nn

    # nn.Identity has no out_features attribute
    bad_extractor = nn.Identity()
    with pytest.raises(ValueError, match="out_features"):
        Classifier(feature_extractor=bad_extractor)


def test_classifier_forward(classifier):
    x = torch.randn(4, 3, 224, 224)
    logits = classifier(x)
    assert logits.shape == (4, 6)


def test_classifier_frozen_backbone(backbone):
    """Backbone parameters must be frozen when freeze_backbone=True."""
    model = Classifier(
        feature_extractor=backbone,
        num_classes=6,
        freeze_backbone=True,
        compile_model=False,
    )
    for param in model.feature_extractor.parameters():
        assert not param.requires_grad


def test_classifier_unfrozen_backbone(backbone):
    """Backbone parameters must be trainable when freeze_backbone=False."""
    model = Classifier(
        feature_extractor=backbone,
        num_classes=6,
        freeze_backbone=False,
        compile_model=False,
    )
    assert any(p.requires_grad for p in model.feature_extractor.parameters())


def test_classifier_configure_optimizers(classifier):
    optimizers = classifier.configure_optimizers()
    assert "optimizer" in optimizers
    assert "lr_scheduler" in optimizers


@pytest.mark.slow
def test_classifier_training_step(classifier):
    batch = (torch.randn(4, 3, 224, 224), torch.randint(0, 6, (4,)))
    loss = classifier.training_step(batch, 0)
    assert loss is not None
    assert loss.item() > 0


@pytest.mark.slow
def test_classifier_validation_step(classifier):
    batch = (torch.randn(4, 3, 224, 224), torch.randint(0, 6, (4,)))
    loss = classifier.validation_step(batch, 0)
    assert loss is not None


@pytest.mark.slow
def test_classifier_test_step(classifier):
    batch = (torch.randn(4, 3, 224, 224), torch.randint(0, 6, (4,)))
    loss = classifier.test_step(batch, 0)
    assert loss is not None
