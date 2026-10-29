"""Run all ML baseline experiments (all classifiers x all feature types)."""

import subprocess
import sys

from jute_disease.models.ml import FEATURE_EXTRACTORS, ML_CLASSIFIERS
from jute_disease.training.logger import get_logger

logger = get_logger(__name__)

CLASSIFIERS_KEYS = list(ML_CLASSIFIERS.keys())
FEATURE_TYPES_KEYS = list(FEATURE_EXTRACTORS.keys())


def run_all_ml(
    classifiers: list[str] = CLASSIFIERS_KEYS,
    feature_types: list[str] = FEATURE_TYPES_KEYS,
    balanced: bool = True,
) -> None:
    """Execute all combinations of classical ML experiments."""
    total = len(classifiers) * len(feature_types)
    logger.info(f"Starting ML Training Pipeline — {total} experiments.")

    failed: list[str] = []
    for feat in feature_types:
        for clf in classifiers:
            logger.info(f"Training {clf} with {feat} features...")

            cmd = [
                sys.executable,
                "-m",
                "jute_disease.cli",
                "train",
                "ml",
                "--classifier",
                clf,
                "--feature_type",
                feat,
            ]

            if balanced:
                cmd.append("--balanced")
            else:
                cmd.append("--no-balanced")

            result = subprocess.run(cmd)
            if result.returncode != 0:
                logger.error(f"{clf}/{feat} failed with exit code {result.returncode}.")
                failed.append(f"{clf}/{feat}")
            else:
                logger.info(f"Finished {clf}/{feat}.")

    if failed:
        raise RuntimeError(f"Failed experiments: {failed}")

    logger.info("All ML experiments completed!")
