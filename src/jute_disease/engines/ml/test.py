# ruff: noqa: N806
import os

import torch
import wandb
from sklearn.metrics import classification_report
from torchvision.datasets import ImageFolder

from jute_disease.data import ml_val_transforms
from jute_disease.models.ml import (
    FEATURE_EXTRACTORS,
    ML_CLASSIFIERS,
    extract_features,
)
from jute_disease.utils import (
    DEFAULT_SEED,
    EVAL_METRICS,
    ML_SPLIT_DIR,
    WANDB_ENTITY,
    WANDB_PROJECT,
    format_metrics,
    get_logger,
    seed_everything,
    setup_wandb,
)

logger = get_logger(__name__)


def test_ml(
    classifier: str = "rf",
    feature_type: str = "crafted",
    seed: int = DEFAULT_SEED,
) -> None:
    seed_everything(seed)

    # 1. Load Model
    classifier_cls = ML_CLASSIFIERS[classifier]
    model = classifier_cls.load(f"{classifier}_{feature_type}")
    if model is None:
        logger.error(f"No saved model found for {classifier}. Please train it first.")
        return

    logger.info(f"Loaded {classifier} for evaluation.")

    # 2. Setup Feature Extractor
    extractor_cls = FEATURE_EXTRACTORS[feature_type]
    extractor = extractor_cls()

    # 3. Load Test Data
    test_dir = ML_SPLIT_DIR / "test"
    if not test_dir.exists():
        logger.error(f"Test directory not found at {test_dir}")
        return

    test_ds = ImageFolder(root=test_dir, transform=ml_val_transforms)
    class_names = test_ds.classes

    # 4. Extract Features
    X_test, y_test = extract_features(test_ds, extractor=extractor, cache_name="test")

    # 5. Predict and Evaluate
    logger.info("Running predictions on test set...")
    y_pred = model.predict(X_test)

    evaluator = EVAL_METRICS.clone()
    test_out = evaluator(torch.tensor(y_pred), torch.tensor(y_test))
    test_metrics = format_metrics(test_out, prefix="test_")

    logger.info(f"Test Accuracy: {test_metrics['test_acc']:.4f}")
    logger.info(f"Test F1 Macro: {test_metrics['test_f1']:.4f}")
    logger.info(
        "\nClassification Report:\n"
        + classification_report(y_test, y_pred, target_names=class_names)
    )

    # 6. Log to WandB
    if os.getenv("WANDB_MODE") != "disabled":
        setup_wandb()
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=f"Eval-ML-{classifier}-{feature_type}",
            job_type="evaluation",
            config={
                "classifier": classifier,
                "feature_type": feature_type,
                "seed": seed,
            },
        )
        wandb_logs = {**test_metrics}
        wandb_logs["test_conf_mat"] = wandb.plot.confusion_matrix(
            preds=y_pred, y_true=y_test, class_names=class_names
        )
        wandb.log(wandb_logs)
        wandb.finish()
