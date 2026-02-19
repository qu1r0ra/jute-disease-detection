# ruff: noqa: N806
import argparse
from pathlib import Path

from PIL import Image

from jute_disease.models.ml import (
    HandcraftedFeatureExtractor,
    KNearestNeighbors,
    LogisticRegression,
    MultinomialNaiveBayes,
    RandomForest,
    RawPixelFeatureExtractor,
    SklearnClassifier,
    SupportVectorMachine,
)
from jute_disease.utils import ML_SPLIT_DIR, get_logger

logger = get_logger(__name__)

ML_CLASSIFIERS: dict[str, type[SklearnClassifier]] = {
    "knn": KNearestNeighbors,
    "lr": LogisticRegression,
    "mnb": MultinomialNaiveBayes,
    "rf": RandomForest,
    "svm": SupportVectorMachine,
}


def predict_ml() -> None:
    parser = argparse.ArgumentParser(description="Jute Classical ML Inference")
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image to classify",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="rf",
        choices=list(ML_CLASSIFIERS.keys()),
        help="Classical ML classifier to use for prediction",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="handcrafted",
        choices=["handcrafted", "raw"],
        help="Type of features the model was trained on",
    )

    args = parser.parse_args()

    # 1. Load Model
    classifier_cls = ML_CLASSIFIERS[args.classifier]
    model = classifier_cls.load()
    if model is None:
        logger.error(
            f"No saved model found for {args.classifier}. Please train it first."
        )
        return

    # 2. Setup Feature Extractor
    if args.feature_type == "handcrafted":
        extractor = HandcraftedFeatureExtractor()
    else:
        extractor = RawPixelFeatureExtractor()

    # 3. Load and Process Image
    img_path = Path(args.image_path)
    if not img_path.exists():
        logger.error(f"Image not found at {img_path}")
        return

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return

    # 4. Extract Features
    logger.info(f"Extracting features from {img_path.name}...")
    feats = extractor.extract(img)
    X = feats.reshape(1, -1)

    # 5. Predict
    prediction = model.predict(X)[0]

    # Try to get class names from the training directory if it exists
    class_names: list[str] | None = None
    train_dir = ML_SPLIT_DIR / "train"
    if train_dir.exists():
        class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

    if class_names and prediction < len(class_names):
        result = class_names[prediction]
        logger.info(f"Prediction: {result} (index: {prediction})")
    else:
        logger.info(f"Prediction: index {prediction}")

    # Optional: Proba
    try:
        proba = model.predict_proba(X)[0]
        if class_names:
            logger.info("Class Probabilities:")
            for name, p in zip(class_names, proba, strict=True):
                logger.info(f"  {name}: {p:.4f}")
        else:
            logger.info(f"Probabilities: {proba}")
    except Exception:
        # Some models might not support predict_proba or might need extra config
        pass


if __name__ == "__main__":
    predict_ml()
