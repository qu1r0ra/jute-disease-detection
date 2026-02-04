from sklearn.naive_bayes import MultinomialNB

from jute_disease_pest.ml.base import BaseMLModel


# TODO: To be implemented
class MultinomialNaiveBayes(BaseMLModel):
    def __init__(self):
        self.model = MultinomialNB()
