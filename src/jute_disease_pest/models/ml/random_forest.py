from sklearn.ensemble import RandomForestClassifier

from jute_disease_pest.ml.base import BaseMLModel


# TODO: To be implemented
class RandomForest(BaseMLModel):
    def __init__(self):
        self.model = RandomForestClassifier()
