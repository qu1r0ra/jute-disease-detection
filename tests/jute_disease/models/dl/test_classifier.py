"""Unit tests for the DL Classifier (LightningModule)."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from jute_disease.models.dl import Classifier, TimmBackbone
from jute_disease.utils.constants import DEFAULT_LR, NUM_CLASSES


@pytest.fixture
def backbone() -> TimmBackbone:
    return TimmBackbone(model_name="resnet18", pretrained=False)


@pytest.fixture
def mock_trainer() -> MagicMock:
    trainer = MagicMock()
    # Mock some expected trainer behaviors
    trainer.should_stop = False
    trainer.global_step = 0
    return trainer


@pytest.fixture
def classifier(backbone: TimmBackbone, mock_trainer: MagicMock) -> Classifier:
    model = Classifier(
        feature_extractor=backbone,
        num_classes=NUM_CLASSES,
        lr=DEFAULT_LR,
        freeze_backbone=False,
        compile_model=False,
    )
    # Attach mock trainer to satisfy self.log calls in steps
    model.trainer = mock_trainer
    return model


def test_classifier_requires_out_features() -> None:
    """Classifier must reject a feature extractor without out_features."""
    # nn.Identity has no out_features attribute
    bad_extractor = nn.Identity()
    with pytest.raises(ValueError, match="out_features"):
        Classifier(feature_extractor=bad_extractor, num_classes=NUM_CLASSES)


def test_classifier_forward(classifier: Classifier) -> None:
    x = torch.randn(4, 3, 224, 224)
    logits = classifier(x)
    assert logits.shape == (4, NUM_CLASSES)


def test_classifier_frozen_backbone(backbone: TimmBackbone) -> None:
    """Backbone parameters must be frozen when freeze_backbone=True."""
    model = Classifier(
        feature_extractor=backbone,
        num_classes=NUM_CLASSES,
        freeze_backbone=True,
        compile_model=False,
    )
    for param in model.feature_extractor.parameters():
        assert not param.requires_grad


def test_classifier_unfrozen_backbone(backbone: TimmBackbone) -> None:
    """Backbone parameters must be trainable when freeze_backbone=False."""
    model = Classifier(
        feature_extractor=backbone,
        num_classes=NUM_CLASSES,
        freeze_backbone=False,
        compile_model=False,
    )
    assert any(p.requires_grad for p in model.feature_extractor.parameters())


def test_classifier_configure_optimizers(classifier: Classifier) -> None:
    optimizers = classifier.configure_optimizers()
    assert "optimizer" in optimizers
    assert "lr_scheduler" in optimizers


def test_classifier_training_step(classifier: Classifier) -> None:
    batch = (torch.randn(4, 3, 224, 224), torch.randint(0, NUM_CLASSES, (4,)))
    loss = classifier.training_step(batch, 0)
    assert loss is not None
    assert loss.item() > 0


def test_classifier_validation_step(classifier: Classifier) -> None:
    batch = (torch.randn(4, 3, 224, 224), torch.randint(0, NUM_CLASSES, (4,)))
    loss = classifier.validation_step(batch, 0)
    assert loss is not None


def test_classifier_test_step(classifier: Classifier) -> None:
    batch = (torch.randn(4, 3, 224, 224), torch.randint(0, NUM_CLASSES, (4,)))
    loss = classifier.test_step(batch, 0)
    assert loss is not None
    assert len(classifier.test_preds) == 1
    assert len(classifier.test_targets) == 1


def test_classifier_predict_step(classifier: Classifier) -> None:
    batch = (torch.randn(4, 3, 224, 224), torch.randint(0, NUM_CLASSES, (4,)))
    preds = classifier.predict_step(batch, 0)
    assert preds.shape == (4, NUM_CLASSES)


def test_classifier_extra_repr(classifier: Classifier) -> None:
    rep = classifier.extra_repr()
    assert f"num_classes={NUM_CLASSES}" in rep
    assert f"lr={DEFAULT_LR}" in rep


def test_classifier_on_test_epoch_hooks(classifier: Classifier) -> None:
    classifier.test_preds.append(torch.tensor([1, 2]))
    classifier.test_targets.append(torch.tensor([1, 2]))

    classifier.on_test_epoch_start()
    assert len(classifier.test_preds) == 0
    assert len(classifier.test_targets) == 0

    classifier.test_preds.append(torch.tensor([1, 2]))
    classifier.test_targets.append(torch.tensor([1, 2]))
    classifier.on_test_epoch_end()
    assert len(classifier.test_preds) == 0
    assert len(classifier.test_targets) == 0
