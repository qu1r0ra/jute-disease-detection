from sklearn.linear_model import LogisticRegression as SKLogisticRegression

from jute_disease_pest.ml.base import BaseMLModel


# TODO: To be implemented
class LogisticRegression(BaseMLModel):
    def __init__(self):
        self.model = SKLogisticRegression()
