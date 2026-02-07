# ruff: noqa: N803
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from jute_disease.models.ml.base import BaseMLModel


class KNearestNeighbors(BaseMLModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = KNeighborsClassifier(**kwargs)

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "KNearestNeighbors":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
