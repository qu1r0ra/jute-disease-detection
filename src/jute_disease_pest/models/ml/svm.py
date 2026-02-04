from sklearn.svm import SVC

from jute_disease_pest.ml.base import BaseMLModel


# TODO: To be implemented
class SVM(BaseMLModel):
    def __init__(self):
        self.model = SVC()
