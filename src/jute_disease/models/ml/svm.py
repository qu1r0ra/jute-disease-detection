# ruff: noqa: N803
import numpy as np
from sklearn.svm import SVC

from jute_disease.models.ml.base import BaseMLModel


class SupportVectorMachine(BaseMLModel):
    def __init__(self, **kwargs):
        super().__init__()
        if "probability" not in kwargs:
            kwargs["probability"] = True
        self.model = SVC(**kwargs)

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "SupportVectorMachine":
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
