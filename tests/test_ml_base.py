import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from jute_disease.models.ml.base import SklearnClassifier


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


class MockNoWeightEstimator(BaseEstimator, ClassifierMixin):
    """Estimator that does not support sample_weight in fit."""

    def fit(self, X, y):
        return self


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
