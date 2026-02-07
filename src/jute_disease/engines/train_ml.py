# ruff: noqa: N806
import argparse
import os

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import f1_score
from torchvision.datasets import ImageFolder

import wandb
from jute_disease.data.transforms import ml_train_transforms, ml_val_transforms
from jute_disease.models.ml.base import BaseMLModel
from jute_disease.models.ml.knn import KNearestNeighbors
from jute_disease.models.ml.logistic_regression import LogisticRegression
from jute_disease.models.ml.naive_bayes import MultinomialNaiveBayes
from jute_disease.models.ml.random_forest import RandomForest
from jute_disease.models.ml.svm import SupportVectorMachine
from jute_disease.utils.constants import (
    DEFAULT_SEED,
    ML_SPLIT_DIR,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from jute_disease.utils.feature_extractor import (
    HandcraftedFeatureExtractor,
    RawPixelFeatureExtractor,
    extract_features,
)
from jute_disease.utils.logger import get_logger
from jute_disease.utils.seed import seed_everything

logger = get_logger(__name__)

ML_CLASSIFIERS: dict[str, BaseMLModel] = {
    "knn": KNearestNeighbors,
    "lr": LogisticRegression,
    "mnb": MultinomialNaiveBayes,
    "rf": RandomForest,
    "svm": SupportVectorMachine,
}


def train_ml():
    parser = argparse.ArgumentParser(description="Jute Classical ML Training")
    parser.add_argument(
        "--classifier",
        type=str,
        default="rf",
        choices=list(ML_CLASSIFIERS.keys()),
        help="Classical ML classifier to train",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="handcrafted",
        choices=["handcrafted", "raw"],
        help="Type of features to extract (handcrafted or raw)",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=True,
        help="Whether to use balanced sample weights during training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed",
    )

    args = parser.parse_args()
    seed_everything(args.seed)
    load_dotenv()

    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        wandb.login()

    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"ClassicalML-{args.classifier}",
        config={
            "classifier": args.classifier,
            "feature_type": args.feature_type,
            "balanced": args.balanced,
            "seed": args.seed,
        },
    )

    if args.feature_type == "handcrafted":
        extractor = HandcraftedFeatureExtractor()
    else:
        extractor = RawPixelFeatureExtractor()

    train_ds = ImageFolder(root=ML_SPLIT_DIR / "train", transform=ml_train_transforms)
    val_ds = ImageFolder(root=ML_SPLIT_DIR / "val", transform=ml_val_transforms)
    test_ds = ImageFolder(root=ML_SPLIT_DIR / "test", transform=ml_val_transforms)
    X_train, y_train = extract_features(train_ds, extractor=extractor)
    X_val, y_val = extract_features(val_ds, extractor=extractor)
    X_test, y_test = extract_features(test_ds, extractor=extractor)

    logger.info(f"Training {args.classifier}...")
    classifier_cls = ML_CLASSIFIERS[args.classifier]
    model = classifier_cls()

    sample_weight = None
    if args.balanced:
        unique_classes, counts = np.unique(y_train, return_counts=True)
        class_weights = len(y_train) / (len(unique_classes) * counts)
        weight_map = dict(zip(unique_classes, class_weights, strict=True))
        sample_weight = np.array([weight_map[label] for label in y_train])
        logger.info("Calculated balanced sample weights for training.")

    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_val_pred = model.predict(X_val)
    acc = np.mean(y_val_pred == y_val)
    f1 = f1_score(y_val, y_val_pred, average="macro")
    logger.info(f"Validation Accuracy: {acc:.4f}")
    logger.info(f"Validation F1 Macro: {f1:.4f}")

    y_test_pred = model.predict(X_test)
    test_acc = np.mean(y_test_pred == y_test)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test F1 Macro: {test_f1:.4f}")

    wandb.log(
        {
            "val_acc": acc,
            "val_f1": f1,
            "test_acc": test_acc,
            "test_f1": test_f1,
        }
    )
    wandb.finish()
    model.save()


if __name__ == "__main__":
    train_ml()
