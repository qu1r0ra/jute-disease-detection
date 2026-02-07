# ruff: noqa: N803
import numpy as np
from sklearn.naive_bayes import MultinomialNB

from jute_disease.models.ml.base import BaseMLModel


class MultinomialNaiveBayes(BaseMLModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = MultinomialNB(**kwargs)

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "MultinomialNaiveBayes":
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
