"""Unit tests for all SklearnClassifier subclasses."""

# ruff: noqa: N803, N806
import numpy as np
import pytest

from jute_disease.models.ml import (
    KNearestNeighbors,
    LogisticRegression,
    MultinomialNaiveBayes,
    RandomForest,
    SupportVectorMachine,
)


@pytest.fixture
def xy():
    """Tiny 2-class dataset with non-negative features (for MNB compatibility)."""
    rng = np.random.default_rng(42)
    X = np.abs(rng.random((30, 8))).astype(np.float32)
    y = np.array([0, 1] * 15)
    return X, y


@pytest.mark.parametrize(
    "cls",
    [LogisticRegression, RandomForest, SupportVectorMachine, MultinomialNaiveBayes],
)
def test_classifier_fit_predict(cls, xy):
    X, y = xy
    model = cls()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (30,)
    assert set(preds).issubset({0, 1})


@pytest.mark.parametrize(
    "cls",
    [LogisticRegression, RandomForest, SupportVectorMachine, MultinomialNaiveBayes],
)
def test_classifier_predict_proba(cls, xy):
    X, y = xy
    model = cls()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (30, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_knn_fit_predict(xy):
    X, y = xy
    model = KNearestNeighbors(n_neighbors=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (30,)


def test_knn_ignores_sample_weight(xy, caplog):
    """KNN does not support sample_weight — must log a warning, not crash."""
    X, y = xy
    sw = np.ones(30)
    model = KNearestNeighbors(n_neighbors=3)
    model.fit(X, y, sample_weight=sw)
    assert "does not support sample_weight" in caplog.text


def test_svm_has_probability_by_default(xy):
    """SVM must have probability=True so predict_proba works without explicit config."""
    X, y = xy
    model = SupportVectorMachine()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape[1] == 2


def test_classifiers_with_sample_weight(xy):
    """Classifiers that support sample_weight must accept it without error."""
    X, y = xy
    sw = np.ones(30)
    for cls in [LogisticRegression, RandomForest, SupportVectorMachine]:
        model = cls()
        model.fit(X, y, sample_weight=sw)


def test_classifier_save_load(tmp_path, monkeypatch, xy):
    from jute_disease.models.ml import classifiers

    monkeypatch.setattr(classifiers, "ML_MODELS_DIR", tmp_path)

    X, y = xy
    model = RandomForest()
    model.fit(X, y)
    model.save()

    assert (tmp_path / "randomforest.joblib").exists()

    loaded = RandomForest.load()
    assert loaded is not None
    preds = loaded.predict(X)
    assert preds.shape == (30,)
