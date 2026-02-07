# ruff: noqa: N803
import os
from abc import ABC, abstractmethod

import joblib
import numpy as np

from jute_disease.utils.constants import ML_MODELS_DIR
from jute_disease.utils.logger import get_logger

logger = get_logger(__name__)


class BaseMLModel(ABC):
    def __init__(self, **kwargs):
        self.model = None

    @abstractmethod
    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "BaseMLModel":
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes for samples in X."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X."""
        pass

    # Model persistence methods can be found at
    # https://scikit-learn.org/stable/model_persistence.html
    def save(self, path: str | None = None):
        """Save the model to disk. If path is None, saves to dump directory."""
        if path is None:
            ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = str(ML_MODELS_DIR / f"{self.__class__.__name__.lower()}.joblib")

        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str | None = None):
        """Load the model from disk. If path is None, loads from dump directory."""
        if path is None:
            path = str(ML_MODELS_DIR / f"{self.__class__.__name__.lower()}.joblib")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")

        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
