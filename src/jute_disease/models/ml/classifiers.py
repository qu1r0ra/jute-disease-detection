# ruff: noqa: N803
"""Scikit-learn classifier wrappers for the jute disease ML pipeline."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from jute_disease.utils.constants import ML_MODELS_DIR
from jute_disease.utils.logger import get_logger

logger = get_logger(__name__)


class SklearnClassifier:
    """Adapter that wraps any scikit-learn classifier into a consistent interface."""

    supports_sample_weight: bool = True

    def __init__(self, sklearn_cls: type[ClassifierMixin], **kwargs: object) -> None:
        if not issubclass(sklearn_cls, ClassifierMixin):
            raise TypeError(f"{sklearn_cls} must be a scikit-learn classifier.")
        self.model = sklearn_cls(**kwargs)

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> None:
        if sample_weight is not None and not self.supports_sample_weight:
            logger.warning(
                f"{self.__class__.__name__} does not support sample_weight. "
                "Weighting will be ignored."
            )
            self.model.fit(X, y)
        elif sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self) -> None:
        ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = str(ML_MODELS_DIR / f"{self.__class__.__name__.lower()}.joblib")
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls) -> "SklearnClassifier | None":
        path = ML_MODELS_DIR / f"{cls.__name__.lower()}.joblib"
        if not Path(path).exists():
            return None
        instance = cls()
        instance.model = joblib.load(str(path))
        return instance


class KNearestNeighbors(SklearnClassifier):
    supports_sample_weight = False

    def __init__(self, **kwargs: object) -> None:
        super().__init__(KNeighborsClassifier, **kwargs)


class SupportVectorMachine(SklearnClassifier):
    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("probability", True)
        super().__init__(SVC, **kwargs)


class LogisticRegression(SklearnClassifier):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(SKLogisticRegression, **kwargs)


class RandomForest(SklearnClassifier):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(RandomForestClassifier, **kwargs)


class MultinomialNaiveBayes(SklearnClassifier):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(MultinomialNB, **kwargs)
