from sklearn.linear_model import SGDClassifier

from jute_disease_pest.ml.base import BaseMLModel


# TODO: To be implemented
class SGD(BaseMLModel):
    def __init__(self):
        self.model = SGDClassifier()
