# ruff: noqa: N806
import argparse
import os

import numpy as np
from sklearn.metrics import classification_report, f1_score
from torchvision.datasets import ImageFolder

import wandb
from jute_disease.data import ml_val_transforms
from jute_disease.models.ml import (
    HandcraftedFeatureExtractor,
    KNearestNeighbors,
    LogisticRegression,
    MultinomialNaiveBayes,
    RandomForest,
    RawPixelFeatureExtractor,
    SklearnClassifier,
    SupportVectorMachine,
    extract_features,
)
from jute_disease.utils import (
    DEFAULT_SEED,
    ML_SPLIT_DIR,
    WANDB_ENTITY,
    WANDB_PROJECT,
    get_logger,
    seed_everything,
    setup_wandb,
)

logger = get_logger(__name__)

ML_CLASSIFIERS: dict[str, type[SklearnClassifier]] = {
    "knn": KNearestNeighbors,
    "lr": LogisticRegression,
    "mnb": MultinomialNaiveBayes,
    "rf": RandomForest,
    "svm": SupportVectorMachine,
}


def test_ml() -> None:
    parser = argparse.ArgumentParser(description="Jute Classical ML Evaluation")
    parser.add_argument(
        "--classifier",
        type=str,
        default="rf",
        choices=list(ML_CLASSIFIERS.keys()),
        help="Classical ML classifier to evaluate",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="handcrafted",
        choices=["handcrafted", "raw"],
        help="Type of features used (handcrafted or raw)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed",
    )

    args = parser.parse_args()
    seed_everything(args.seed)

    # 1. Load Model
    classifier_cls = ML_CLASSIFIERS[args.classifier]
    model = classifier_cls.load()
    if model is None:
        logger.error(
            f"No saved model found for {args.classifier}. Please train it first."
        )
        return

    logger.info(f"Loaded {args.classifier} for evaluation.")

    # 2. Setup Feature Extractor
    if args.feature_type == "handcrafted":
        extractor = HandcraftedFeatureExtractor()
    else:
        extractor = RawPixelFeatureExtractor()

    # 3. Load Test Data
    test_dir = ML_SPLIT_DIR / "test"
    if not test_dir.exists():
        logger.error(f"Test directory not found at {test_dir}")
        return

    test_ds = ImageFolder(root=test_dir, transform=ml_val_transforms)
    class_names = test_ds.classes

    # 4. Extract Features
    X_test, y_test = extract_features(test_ds, extractor=extractor)

    # 5. Predict and Evaluate
    logger.info("Running predictions on test set...")
    y_pred = model.predict(X_test)

    acc = float(np.mean(y_pred == y_test))
    f1 = float(f1_score(y_test, y_pred, average="macro"))

    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Test F1 Macro: {f1:.4f}")
    logger.info(
        "\nClassification Report:\n"
        + classification_report(y_test, y_pred, target_names=class_names)
    )

    # 6. WandB (Optional)
    if os.environ.get("WANDB_MODE") != "disabled":
        setup_wandb()
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=f"Eval-ML-{args.classifier}",
            job_type="evaluation",
            config=vars(args),
        )
        wandb.log(
            {
                "test_acc": acc,
                "test_f1": f1,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    test_ml()
