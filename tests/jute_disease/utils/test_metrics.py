import pytest
import torch
from torchmetrics import MetricCollection

from jute_disease.utils.constants import NUM_CLASSES
from jute_disease.utils.metrics import EVAL_METRICS, TRAIN_METRICS, format_metrics


def test_format_metrics_no_prefix() -> None:
    # Setup mock metric dictionary directly with tensors
    mock_metrics = {
        "acc": torch.tensor(0.95),
        "f1": torch.tensor(0.94),
    }

    formatted = format_metrics(mock_metrics)

    assert isinstance(formatted, dict)
    assert len(formatted) == 2
    assert formatted["acc"] == pytest.approx(0.95)
    assert formatted["f1"] == pytest.approx(0.94)
    assert all(isinstance(v, float) for v in formatted.values())


def test_format_metrics_with_prefix() -> None:
    # Setup mock metric dictionary
    mock_metrics = {
        "acc": torch.tensor(0.85),
        "recall": torch.tensor(0.82),
    }

    formatted = format_metrics(mock_metrics, prefix="test_")

    assert "test_acc" in formatted
    assert "test_recall" in formatted
    assert formatted["test_acc"] == pytest.approx(0.85)
    assert formatted["test_recall"] == pytest.approx(0.82)


def test_train_metrics_structure() -> None:
    assert isinstance(TRAIN_METRICS, MetricCollection)
    assert "acc" in TRAIN_METRICS
    # Test num_classes config
    assert TRAIN_METRICS["acc"].num_classes == NUM_CLASSES


def test_eval_metrics_structure() -> None:
    assert isinstance(EVAL_METRICS, MetricCollection)
    assert "acc" in EVAL_METRICS
    assert "f1" in EVAL_METRICS
    assert "precision" in EVAL_METRICS
    assert "recall" in EVAL_METRICS

    # Test configurations
    assert EVAL_METRICS["f1"].num_classes == NUM_CLASSES
    assert EVAL_METRICS["f1"].average == "macro"
