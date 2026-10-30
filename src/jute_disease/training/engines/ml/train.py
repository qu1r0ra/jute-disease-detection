import os

import numpy as np
import torch
import wandb

from jute_disease.config import DEFAULT_SEED, WANDB_ENTITY, WANDB_PROJECT
from jute_disease.data import classical_pipeline, open_partition
from jute_disease.evaluation.metrics import EVAL_METRICS, format_metrics
from jute_disease.models.ml import (
    FEATURE_EXTRACTORS,
    ML_CLASSIFIERS,
    extract_features,
)
from jute_disease.training.logger import get_logger, setup_wandb
from jute_disease.training.seed import seed_everything

logger = get_logger(__name__)


def train_ml(
    classifier: str = "rf",
    feature_type: str = "crafted",
    balanced: bool = True,
    seed: int = DEFAULT_SEED,
) -> None:
    seed_everything(seed)

    if os.getenv("WANDB_MODE") != "disabled":
        setup_wandb()

        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=f"{classifier}-{feature_type}",
            config={
                "classifier": classifier,
                "feature_type": feature_type,
                "balanced": balanced,
                "seed": seed,
            },
        )

    extractor_cls = FEATURE_EXTRACTORS[feature_type]
    extractor = extractor_cls()

    # Classical features use the same deterministic preprocessing in every
    # partition; the frozen manifest alone decides which image is where.
    train_ds, val_ds, test_ds = (
        open_partition(name, classical_pipeline()) for name in ("train", "val", "test")
    )

    X_train, y_train = extract_features(train_ds, extractor=extractor)
    X_val, y_val = extract_features(val_ds, extractor=extractor)
    X_test, y_test = extract_features(test_ds, extractor=extractor)

    logger.info(f"Training {classifier}...")
    classifier_cls = ML_CLASSIFIERS[classifier]
    model = classifier_cls(random_state=seed)

    sample_weight = None
    if balanced:
        unique_classes, counts = np.unique(y_train, return_counts=True)
        class_weights = len(y_train) / (len(unique_classes) * counts)
        weight_map = dict(zip(unique_classes, class_weights, strict=True))
        sample_weight = np.array([weight_map[label] for label in y_train])
        logger.info("Calculated balanced sample weights for training.")

    evaluator = EVAL_METRICS.clone()

    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_val_pred = model.predict(X_val)
    val_out = evaluator(torch.tensor(y_val_pred), torch.tensor(y_val))
    val_metrics = format_metrics(val_out, prefix="val_")

    logger.info(f"Validation Accuracy: {val_metrics['val_acc']:.4f}")
    logger.info(f"Validation F1 Macro: {val_metrics['val_f1']:.4f}")

    y_test_pred = model.predict(X_test)
    test_out = evaluator(torch.tensor(y_test_pred), torch.tensor(y_test))
    test_metrics = format_metrics(test_out, prefix="test_")

    logger.info(f"Test Accuracy: {test_metrics['test_acc']:.4f}")
    logger.info(f"Test F1 Macro: {test_metrics['test_f1']:.4f}")

    if os.getenv("WANDB_MODE") != "disabled":
        class_names = test_ds.classes
        wandb_logs = {**val_metrics, **test_metrics}
        wandb_logs["test_conf_mat"] = wandb.plot.confusion_matrix(
            preds=y_test_pred, y_true=y_test, class_names=class_names
        )
        wandb.log(wandb_logs)
        wandb.finish()
    model.save(f"{classifier}_{feature_type}")
