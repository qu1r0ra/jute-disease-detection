# ruff: noqa: N803, N806
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from jute_disease.models.ml import SklearnClassifier


class MockEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, check_sample_weight=False):
        self.check_sample_weight = check_sample_weight
        self.sample_weight_passed = None

    def fit(self, X, y, sample_weight=None):
        self.sample_weight_passed = sample_weight
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.zeros((len(X), 2))


def test_sklearn_classifier_adapter_passes_weight():
    X = np.random.rand(10, 2)
    y = np.array([0, 1] * 5)
    sw = np.random.rand(10)

    adapter = SklearnClassifier(MockEstimator)
    adapter.fit(X, y, sample_weight=sw)

    assert adapter.model.sample_weight_passed is not None
    assert np.array_equal(adapter.model.sample_weight_passed, sw)


def test_sklearn_classifier_warns_no_weight(caplog):
    X = np.random.rand(10, 2)
    y = np.array([0, 1] * 5)
    sw = np.random.rand(10)

    # Force supports_sample_weight to False for testing warning
    class NoWeightAdapter(SklearnClassifier):
        supports_sample_weight = False

    adapter = NoWeightAdapter(MockEstimator)
    adapter.fit(X, y, sample_weight=sw)

    assert "does not support sample_weight" in caplog.text
    assert adapter.model.sample_weight_passed is None


def test_sklearn_classifier_predict():
    X = np.random.rand(10, 2)
    adapter = SklearnClassifier(MockEstimator)
    adapter.model = MockEstimator()

    y_pred = adapter.predict(X)
    assert len(y_pred) == 10
    assert isinstance(y_pred, np.ndarray)


def test_sklearn_classifier_save_load(tmp_path, monkeypatch):
    from jute_disease.models.ml import classifiers

    monkeypatch.setattr(classifiers, "ML_MODELS_DIR", tmp_path)

    X = np.random.rand(10, 2)
    y = np.array([0, 1] * 5)

    class PersistentClassifier(SklearnClassifier):
        def __init__(self, **kwargs):
            super().__init__(MockEstimator, **kwargs)

    adapter = PersistentClassifier()
    adapter.fit(X, y)
    adapter.save()

    # Verify file exists (lowercase class name)
    expected_path = tmp_path / "persistentclassifier.joblib"
    assert expected_path.exists()

    # Load and verify
    loaded = PersistentClassifier.load()
    assert loaded is not None
    assert isinstance(loaded.model, MockEstimator)
    # Check if fit attributes survived
    assert loaded.model.sample_weight_passed is None
